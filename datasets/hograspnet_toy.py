import os, sys
sys.path.append(os.getcwd())

from torch.utils.data import Dataset

import os
import re
# Counter: label별 sample 개수를 세기 위한 dict 형태의 자료구조
# defaultdict: key가 처음 등장해도 자동으로 기본값(list 또는 Counter)을 만들어주는 dict
from collections import Counter, defaultdict

import torch
import numpy as np
from einops import rearrange
import torchvision.transforms as T
import random
import pickle

from datasets.augmentation import RandomYawRotation, RandomJitter

class HOGraspNetToyDataset(Dataset):
    def __init__(self,
                 root,
                 split='train',
                 setup='full',
                 test_ratio=0.15,
                 split_seed=42,
                 split_trials=100,
                 transform=None):
        super().__init__()

        self.root = root
        self.data_li = []
        self.train_data_li, self.test_data_li = [], []
        # test_ratio: 전체 데이터 중 test로 보내고 싶은 비율
        # setup:
        #   "full"이면 full train/test를 그대로 사용한다.
        #   "uniform"이면 train만 label-balanced subset으로 줄이고, test는 full과 동일하게 둔다.
        # split_seed: subject split을 재현 가능하게 만들기 위한 seed
        # split_trials: 여러 초기 split 후보를 만들어 보고 그중 가장 좋은 split을 선택하기 위한 반복 횟수
        self.setup = setup
        self.test_ratio = test_ratio
        self.split_seed = split_seed
        self.split_trials = split_trials
        self.transform = transform

        # NOTE: 이건 이전 setup
        # self._collect_toy_setup()
        # self._select_split()

        self._collect_toy_setup(test_ratio=test_ratio, setup=setup)
        if split == 'train':
            self.data_li = self.train_data_li
        elif split == 'test':
            self.data_li = self.test_data_li
        else:
            raise Exception(f"Check split args, given {split}")

        # Full train set (test_ratio = 0.15)의 std (이때, std는 mean으로 centering을 하고 나서 구하였다.)
        # 또한, 각 축별로 따로 하지 않고 하나의 scalar를 통해서 하는 이유는 3D geometry의 비율을 보존하기 위해서이다.
        # -> 각 축을 따로 하면 x, y, z 축 간 비율이 달라짐 (왜곡됨)
        self.std = 6.5115968138095175

    def __len__(self):
        return len(self.data_li)

    def _parse_subject_id(self, dirname):
        # 예: "231008_S69_obj_25_grasp_5" -> "S69"
        # subject id를 기준으로 train/test를 나누기 위해 directory name에서 subject id를 파싱한다.
        match = re.search(r'_(S\d+)_', dirname)
        if match is None:
            raise ValueError(f"Cannot parse subject id from {dirname}")
        return match.group(1)

    def _split_score(self, test_label_cnt, total_label_cnt, target_ratio):
        # subject 단위로 split하면 label별 test 비율이 정확히 target_ratio가 되기는 어렵다.
        # 따라서 "현재 test subject 집합이 얼마나 target_ratio에 가까운가"를 score로 계산한다.
        # score가 작을수록 좋은 split이다.
        labels = sorted(total_label_cnt.keys())
        label_error = 0.0
        for label in labels:
            # 특정 label 안에서 test sample이 차지하는 비율
            # 예: label 12 전체가 5657개이고 test에 1130개가 있으면 observed_ratio ~= 0.2
            observed_ratio = test_label_cnt[label] / total_label_cnt[label]
            label_error += (observed_ratio - target_ratio) ** 2
        label_error /= max(len(labels), 1)

        # label별 비율뿐 아니라 전체 test sample 수 자체도 target_ratio에 가까워야 한다.
        test_count = sum(test_label_cnt.values())
        total_count = sum(total_label_cnt.values())
        size_error = ((test_count / total_count) - target_ratio) ** 2

        return label_error + size_error

    def _select_subject_disjoint_split(self, subject_label_cnt, total_label_cnt, target_ratio):
        # subject_label_cnt:
        #   subject_label_cnt["S69"][5] = S69 subject에서 label 5 sample 개수
        # total_label_cnt:
        #   total_label_cnt[5] = 전체 데이터에서 label 5 sample 개수
        subjects = list(subject_label_cnt.keys())

        def get_test_label_cnt(test_subjects):
            # 현재 test로 선택된 subject들의 label별 sample 개수를 합산한다.
            test_label_cnt = Counter()
            for subject_id in test_subjects:
                test_label_cnt.update(subject_label_cnt[subject_id])
            return test_label_cnt

        def improve_split(test_subjects):
            # greedy로 만든 초기 split에서 subject 하나를 test에 넣거나 빼는 것을 반복한다.
            # 이때 score가 더 좋아지는 변경만 채택한다.
            # 즉, local search 방식으로 split을 조금씩 개선한다.
            test_subjects = set(test_subjects)
            best_score = self._split_score(get_test_label_cnt(test_subjects), total_label_cnt, target_ratio)

            improved = True
            while improved:
                improved = False
                best_subject = None

                for subject_id in subjects:
                    # candidate는 현재 test subject 집합에서 subject 하나만 토글한 split이다.
                    # test에 있던 subject면 제거하고, train에 있던 subject면 test에 추가한다.
                    candidate = set(test_subjects)
                    if subject_id in candidate:
                        candidate.remove(subject_id)
                    else:
                        candidate.add(subject_id)

                    # 이 변경이 label 비율과 전체 test 비율을 더 target_ratio에 가깝게 만드는지 확인한다.
                    candidate_score = self._split_score(get_test_label_cnt(candidate), total_label_cnt, target_ratio)
                    if candidate_score < best_score:
                        best_score = candidate_score
                        best_subject = subject_id
                        improved = True

                if improved:
                    # 이번 반복에서 가장 score를 낮춘 subject 토글을 실제 split에 반영한다.
                    if best_subject in test_subjects:
                        test_subjects.remove(best_subject)
                    else:
                        test_subjects.add(best_subject)

            return test_subjects, best_score

        best_test_subjects = None
        best_score = float("inf")
        total_count = sum(total_label_cnt.values())

        if self.split_trials < 1:
            raise ValueError(f"split_trials must be at least 1, given {self.split_trials}")

        for trial_idx in range(self.split_trials):
            # split_trials만큼 서로 다른 초기 subject 순서를 만들어 본다.
            # 같은 split_seed를 쓰면 항상 같은 split이 재현된다.
            rng = random.Random(self.split_seed + trial_idx)
            shuffled_subjects = subjects[:]
            rng.shuffle(shuffled_subjects)

            # sample 수가 큰 subject부터 test 후보로 검토한다.
            # 큰 subject는 split 비율에 미치는 영향이 크기 때문에 먼저 배치하는 편이 greedy 초기화에 유리하다.
            # 두 번째 key인 rng.random()은 sample 수가 같은 subject들의 순서를 seed 기반으로 섞기 위한 tie-breaker이다.
            shuffled_subjects.sort(
                key=lambda subject_id: (
                    sum(subject_label_cnt[subject_id].values()),
                    rng.random(),
                ),
                reverse=True,
            )

            test_subjects = set()
            test_label_cnt = Counter()
            for subject_id in shuffled_subjects:
                # 이 subject를 test에 추가했을 때의 label별 test count
                candidate_label_cnt = test_label_cnt + subject_label_cnt[subject_id]

                # 아직 전체 test sample 수가 목표보다 적으면 우선 test에 더 넣는다.
                current_is_short = sum(test_label_cnt.values()) < target_ratio * total_count

                # 또는 이 subject를 추가하는 것이 score를 낮추면 test에 넣는다.
                candidate_is_better = (
                    self._split_score(candidate_label_cnt, total_label_cnt, target_ratio)
                    <= self._split_score(test_label_cnt, total_label_cnt, target_ratio)
                )

                if current_is_short or candidate_is_better:
                    test_subjects.add(subject_id)
                    test_label_cnt = candidate_label_cnt

            # greedy 초기 split을 local search로 한 번 더 개선한다.
            test_subjects, score = improve_split(test_subjects)
            if score < best_score:
                best_score = score
                best_test_subjects = test_subjects

        # test로 선택되지 않은 subject는 모두 train으로 간다.
        # 따라서 train/test 간 subject overlap은 구조적으로 생기지 않는다.
        train_subjects = set(subjects) - best_test_subjects
        return train_subjects, best_test_subjects

    def _make_uniform_train_subset(self, train_data_li):
        # full train set 안에서만 label별 sample 수를 맞춘다.
        # 이렇게 하면 uniform train은 항상 full train의 subset이고,
        # full/uniform setup은 완전히 동일한 full test set을 공유한다.
        data_per_label = defaultdict(list)
        for file_path, label in train_data_li:
            data_per_label[label].append([file_path, label])

        if len(data_per_label) == 0:
            raise ValueError("Cannot make uniform train subset from empty train_data_li")

        # train에 존재하는 label들 중 가장 적은 sample 수를 기준으로 모든 label을 downsample한다.
        # 예: full train에서 label별 count의 최소가 372라면, uniform train은 label마다 372개씩 사용한다.
        min_count = min(len(samples) for samples in data_per_label.values())

        rng = random.Random(self.split_seed)
        uniform_train_data_li = []
        for label in sorted(data_per_label.keys()):
            label_samples = data_per_label[label][:]
            rng.shuffle(label_samples)
            uniform_train_data_li.extend(label_samples[:min_count])

        rng.shuffle(uniform_train_data_li)
        return uniform_train_data_li

    def _collect_toy_setup(self, test_ratio=0.2, setup='full'):
        if not 0.0 < test_ratio < 1.0:
            raise ValueError(f"test_ratio must be between 0 and 1, given {test_ratio}")

        if setup not in ['full', 'uniform']:
            raise ValueError(f"setup must be either 'full' or 'uniform', given {setup}")

        os.makedirs('cache', exist_ok=True)
        ratio_name = str(test_ratio).replace('.', 'p')
        cache_prefix = f"toy_hograspnet_full_ratio_{ratio_name}_seed_{self.split_seed}_trials_{self.split_trials}"
        full_train_cache_path = os.path.join('cache', f'{cache_prefix}_train.pkl')
        full_test_cache_path = os.path.join('cache', f'{cache_prefix}_test.pkl')

        # full과 uniform은 같은 subject-disjoint split을 공유해야 하므로,
        # cache도 항상 full split 기준으로 저장하고 불러온다.
        if os.path.exists(full_train_cache_path) and os.path.exists(full_test_cache_path):
            with open(full_train_cache_path, 'rb') as f:
                full_train_data_li = pickle.load(f)
            with open(full_test_cache_path, 'rb') as f:
                full_test_data_li = pickle.load(f)

            self.train_data_li = full_train_data_li
            self.test_data_li = full_test_data_li
            if setup == 'uniform':
                self.train_data_li = self._make_uniform_train_subset(full_train_data_li)
            return 
        
        
        # subject_data:
        #   subject별 실제 sample path와 label을 저장한다.
        # subject_label_cnt:
        #   subject별 label histogram을 저장한다. split 최적화에 사용된다.
        # total_label_cnt:
        #   전체 데이터의 label histogram을 저장한다. target_ratio 계산 기준이다.
        subject_data = defaultdict(list)
        subject_label_cnt = defaultdict(Counter)
        total_label_cnt = Counter()

        for subject_grasp_dirname in sorted(os.listdir(self.root)):
            subject_grasp_path = os.path.join(self.root, subject_grasp_dirname)

            if not os.path.isdir(subject_grasp_path):
                continue

            # 현재 파일명 규칙에서는 마지막 token을 taxonomy label로 사용한다.
            # 예: "231008_S69_obj_25_grasp_5" -> label 5
            taxonomy_label = int(subject_grasp_dirname.split('_')[-1])
            subject_id = self._parse_subject_id(subject_grasp_dirname)

            file_paths = [
                os.path.join(subject_grasp_path, filename)
                for filename in sorted(os.listdir(subject_grasp_path))
            ]

            file_paths_with_taxo_labels = [[file_path, taxonomy_label] for file_path in file_paths]

            # 실제 데이터는 subject별로 모아두고,
            # 동시에 split을 위한 subject별/전체 label count를 누적한다.
            subject_data[subject_id].extend(file_paths_with_taxo_labels)
            subject_label_cnt[subject_id][taxonomy_label] += len(file_paths)
            total_label_cnt[taxonomy_label] += len(file_paths)

        # subject 단위로 train/test split을 선택한다.
        # 이 함수가 반환하는 train_subjects와 test_subjects는 서로 겹치지 않는다.
        train_subjects, test_subjects = self._select_subject_disjoint_split(
            subject_label_cnt=subject_label_cnt,
            total_label_cnt=total_label_cnt,
            target_ratio=test_ratio,
        )

        # 선택된 subject 집합을 실제 sample list로 변환한다.
        for subject_id in sorted(train_subjects):
            self.train_data_li.extend(subject_data[subject_id])
        for subject_id in sorted(test_subjects):
            self.test_data_li.extend(subject_data[subject_id])

        # split 자체는 subject 기준으로 고정하고, 학습 시 sample 순서만 seed 기반으로 섞는다.
        rng = random.Random(self.split_seed)
        rng.shuffle(self.train_data_li)
        rng.shuffle(self.test_data_li)

        with open(full_train_cache_path, 'wb') as f:
            pickle.dump(self.train_data_li, f)
        with open(full_test_cache_path, 'wb') as f:
            pickle.dump(self.test_data_li, f)

        if setup == 'uniform':
            self.train_data_li = self._make_uniform_train_subset(self.train_data_li)
    
    def __getitem__(self, idx):
        data_path, label = self.data_li[idx]

        sample = np.load(data_path)
        label = int(label)

        hand_pc = sample['hand_pc']
        obj_pc = sample['object_pc']

        total_pc = np.concatenate((hand_pc, obj_pc), axis=0)

        # To Tensor
        total_pc = torch.tensor(total_pc, dtype=torch.float32)

        # Normalization
        total_pc = total_pc - total_pc.mean(axis=0, keepdims=True)
        total_pc = total_pc / self.std

        if self.transform is not None:
            total_pc = self.transform(total_pc)

        if total_pc.dim() == 2: # Default Augmentations (w/o ContrastiveTransformations)
            total_pc = rearrange(total_pc, 'n c -> c n')
        elif total_pc.dim() == 3: # ContrastiveTransformations
            total_pc = rearrange(total_pc, 'b n c -> b c n')

        return total_pc, label

if __name__ == '__main__':
    root = "/workspace/lab_intern/KDW/GraspRep/data/densehograspnet_pointcloud"

    # Taxonomy label statistics
    '''
    taxo_upper_level_cnt = [0 for _ in range(35)]
    taxo_lower_level_cnt = [0 for _ in range(35)]

    for idx, dir_name in enumerate(os.listdir(root), start=1):
        # print(os.path.join(root, dir_name))
        dir_path = os.path.join(root, dir_name)
        taxo_label = int(dir_path.split('_')[-1])
        
        taxo_upper_level_cnt[taxo_label] += 1
        taxo_lower_level_cnt[taxo_label] += len(os.listdir(dir_path))

        temp = os.listdir(os.path.join(root, dir_name))
    
    print(f"count per subject {sum(taxo_upper_level_cnt)}")
    print(f"count per trial {sum(taxo_lower_level_cnt)}")

    for idx, elem in enumerate(taxo_upper_level_cnt):
        if elem == 0: del taxo_upper_level_cnt[idx]

    for idx, elem in enumerate(taxo_lower_level_cnt):
        if elem == 0: del taxo_lower_level_cnt[idx]

    print(f"Number of Taxonomy categories: {len(taxo_lower_level_cnt)}")

    print(f"(min) count per subject {min(taxo_upper_level_cnt)}")
    print(f"(max) count per subject {max(taxo_upper_level_cnt)}")
    
    print(f"(min) count per trial + camera {min(taxo_lower_level_cnt)}")
    print(f"(max) count per trial + camera {max(taxo_lower_level_cnt)}")

    # 여기 세팅 중요
    print(f"Uniform setting: data count {min(taxo_lower_level_cnt) * len(taxo_lower_level_cnt)}")
    print(f"Using all data: {sum(taxo_lower_level_cnt)}")

    import sys; sys.exit()
    '''

    # Data split
    TEST_RATIO = 0.15

    train_ds_full = HOGraspNetToyDataset(
        root = root,
        split = 'train',
        setup = 'full',
        test_ratio = TEST_RATIO,
        split_seed = 42,
        split_trials = 100,
        transform = T.Compose([
            RandomYawRotation(), 
            RandomJitter()
        ])
    )
    test_ds_full = HOGraspNetToyDataset(
        root = root,
        split = 'test',
        setup = 'full',
        test_ratio = TEST_RATIO,
        split_seed = 42,
        split_trials = 100,
        transform = T.Compose([
            RandomYawRotation(), 
            RandomJitter()
        ])
    )

    train_ds_uniform = HOGraspNetToyDataset(
        root = root,
        split = 'train',
        setup = 'uniform',
        test_ratio = TEST_RATIO,
        split_seed = 42,
        split_trials = 100,
        transform = T.Compose([
            RandomYawRotation(), 
            RandomJitter()
        ])
    )

    test_ds_uniform = HOGraspNetToyDataset(
        root = root,
        split = 'test',
        setup = 'uniform',
        test_ratio = TEST_RATIO,
        split_seed = 42,
        split_trials = 100,
        transform = T.Compose([
            RandomYawRotation(), 
            RandomJitter()
        ])
    )

    print(len(train_ds_full))
    print(len(test_ds_full))
    print(len(train_ds_uniform))
    print(len(test_ds_uniform))

    # Data statistics
    '''
    import matplotlib.pyplot as plt

    os.makedirs("assets/", exist_ok=True)

    count = 0
    mean = []
    subject_id_freq_1 = [0 for _ in range(35)]
    subject_id_freq_2 = [0 for _ in range(35)]

    for name in os.listdir(root):
        first_path = os.path.join(root, name)

        class_id = int(first_path.split('_')[-1])

        # root 바로 아래 항목 중 디렉터리만 확인
        if os.path.isdir(first_path):
            num_pointclouds = len(os.listdir(first_path))
            count += num_pointclouds
            mean.append(num_pointclouds)
            subject_id_freq_1[class_id] += 1
            subject_id_freq_2[class_id] += num_pointclouds

    print(f"Subject: {len(os.listdir(root))}")
    print(f"Entire: {count}")
    mean = np.array(mean)
    print(f"Mean of number of PCs for each subject: {(sum(mean)/len(mean))}")
    print(f"Std of number of PCs for each subject: {np.std(mean)}")

    class_ids = np.arange(len(subject_id_freq_1))

    plt.figure(figsize=(12, 5))
    plt.bar(class_ids, subject_id_freq_1)
    plt.xlabel("Class ID")
    plt.ylabel("Count")
    plt.title("Class ID Frequency 1")
    plt.xticks(class_ids)
    plt.tight_layout()
    plt.savefig("assets/subject_id_freq_1_histogram.png", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.bar(class_ids, subject_id_freq_2)
    plt.xlabel("Class ID")
    plt.ylabel("Count")
    plt.title("Class ID Frequency 2")
    plt.xticks(class_ids)
    plt.tight_layout()
    plt.savefig("assets/subject_id_freq_2_histogram.png", dpi=300)
    plt.close()

    width = 0.4
    plt.figure(figsize=(12, 5))
    plt.bar(class_ids - width / 2, subject_id_freq_1, width=width, label="subject_id_freq_1")
    plt.bar(class_ids + width / 2, subject_id_freq_2, width=width, label="subject_id_freq_2")
    plt.xlabel("Class ID")
    plt.ylabel("Count")
    plt.title("Class ID Frequency Histogram")
    plt.xticks(class_ids)
    plt.legend()
    plt.tight_layout()
    plt.savefig("assets/subject_id_freq_combined_histogram.png", dpi=300)
    plt.close()

    print("Saved histograms:")
    print("  assets/subject_id_freq_1_histogram.png")
    print("  assets/subject_id_freq_2_histogram.png")
    print("  assets/subject_id_freq_combined_histogram.png")
    '''

    '''
    ds = DenseHOGraspNetToyDataset(
        root="/workspace/lab_intern/KDW/GraspRep/data/densehograspnet_pointcloud",
        split='train',
        transform=T.Compose([
            RandomYawRotation(), 
            RandomJitter()
        ])
    )
    '''
