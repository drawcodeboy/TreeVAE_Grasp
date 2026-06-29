import random
import re
from collections import Counter, defaultdict
from pathlib import Path
import pickle
import os
import numpy as np

from torch.utils.data import Dataset
from einops import rearrange

class HOGraspNetMANOContactDataset(Dataset):
    FILENAME_PATTERN = re.compile(
        r"^subject_(?P<subject>S\d+)_.*_taxoID_(?P<label>\d+)\.npz$"
    )

    def __init__(
        self,
        root="/workspace/dwkwon/HOGraspNet/processed_data/hand_pose_plus_mano_contact",
        split='train',
        transform=None
    ):
        super().__init__()
        self.root = Path(root)

        # 이 정렬 순서가 split index의 기준이 되는 global index이다.
        self.data_li = sorted(str(path) for path in self.root.glob("*.npz"))
        self.train_indices = []
        self.test_indices = []
        self.split = split

        self.find_uniform_subject_disjoint_indices()

        if self.split == 'train':
            self.data_li = self.train_indices
        elif self.split == 'test':
            self.data_li = self.test_indices

        self.MANO_GLOBAL_SCALE = 5.633076949477
        self.transform = transform

    def __len__(self):
        return len(self.data_li)
    
    def __getitem__(self, idx):
        sample_path = self.data_li[idx]
        data_dict = np.load(sample_path)

        mano_vertices = data_dict['mano_vertices']
        contact_map = data_dict['contact_map']
        taxo_label = int(data_dict['taxo_id'])

        # Normalization
        # Location
        root = data_dict['hand_pose'][0] # wrist joint
        mano_vertices = mano_vertices - root[None, :]
        # Scale
        mano_vertices = mano_vertices / self.MANO_GLOBAL_SCALE

        if self.transform is not None:
            mano_vertices = self.transform(mano_vertices)

        if mano_vertices.dim() == 2: # Default Augmentations (w/o ContrastiveTransformations)
            mano_vertices = rearrange(mano_vertices, 'n c -> c n')
        elif mano_vertices.dim() == 3: # ContrastiveTransformations
            mano_vertices = rearrange(mano_vertices, 'b n c -> b c n')
        
        return (mano_vertices, contact_map), taxo_label

    def _parse_index_metadata(self):
        """Return index-to-subject/taxonomy metadata parsed from filenames."""
        metadata = []
        invalid_filenames = []

        for index, file_path in enumerate(self.data_li):
            filename = Path(file_path).name
            match = self.FILENAME_PATTERN.match(filename)
            if match is None:
                invalid_filenames.append(filename)
                continue
            metadata.append(
                {
                    "index": index,
                    "subject": match.group("subject"),
                    "label": int(match.group("label")),
                }
            )

        if invalid_filenames:
            raise ValueError(
                f"Cannot parse {len(invalid_filenames)} filenames; "
                f"examples: {invalid_filenames[:5]}"
            )
        if not metadata:
            raise ValueError(f"No .npz samples found under {self.root}")
        return metadata

    def find_uniform_subject_disjoint_indices(
        self,
        train_per_label=383,
        test_per_label=100,
        seed=42,
        search_trials=50_000,
    ):
        """Find uniform train/test indices with no subject overlap.

        Despite their names, train_indices and test_indices contain selected
        .npz file paths rather than integer positions in self.data_li.
        """
        train_cache_path = "cache/hograspnet_balanced_train.pkl"
        test_cache_path = "cache/hograspnet_balanced_test.pkl"

        if os.path.exists(train_cache_path) and os.path.exists(test_cache_path):
            with open(train_cache_path, 'rb') as f:
                self.train_indices = pickle.load(f)
            with open(test_cache_path, 'rb') as f:
                self.test_indices = pickle.load(f)

            return self.train_indices, self.test_indices

        if train_per_label <= 0 or test_per_label <= 0:
            raise ValueError("Per-label sample counts must be positive")
        if search_trials <= 0:
            raise ValueError("search_trials must be positive")

        metadata = self._parse_index_metadata()
        subject_label_counts = defaultdict(Counter)
        total_label_counts = Counter()
        for sample in metadata:
            subject_label_counts[sample["subject"]][sample["label"]] += 1
            total_label_counts[sample["label"]] += 1

        labels = sorted(total_label_counts)
        subjects = sorted(subject_label_counts)
        required = train_per_label + test_per_label
        insufficient = {
            label: count
            for label, count in total_label_counts.items()
            if count < required
        }
        if insufficient:
            raise ValueError(
                f"Labels with fewer than {required} samples: {insufficient}"
            )

        rng = random.Random(seed)
        expected_test_size = round(len(subjects) * test_per_label / required)
        candidate_sizes = sorted(
            range(
                max(1, expected_test_size - 5),
                min(len(subjects), expected_test_size + 5) + 1,
            ),
            key=lambda size: abs(size - expected_test_size),
        )
        trials_per_size = max(1, search_trials // len(candidate_sizes))

        selected_partition = None
        best_capacities = (0, 0)
        best_score = -1.0
        for test_size in candidate_sizes:
            for _ in range(trials_per_size):
                test_subjects = frozenset(rng.sample(subjects, test_size))
                test_available = {
                    label: sum(
                        subject_label_counts[subject][label]
                        for subject in test_subjects
                    )
                    for label in labels
                }
                train_available = {
                    label: total_label_counts[label] - test_available[label]
                    for label in labels
                }
                capacities = (
                    min(train_available.values()),
                    min(test_available.values()),
                )
                score = min(
                    capacities[0] / train_per_label,
                    capacities[1] / test_per_label,
                )
                if score > best_score:
                    best_score = score
                    best_capacities = capacities

                if (
                    capacities[0] >= train_per_label
                    and capacities[1] >= test_per_label
                ):
                    selected_partition = (
                        frozenset(set(subjects) - set(test_subjects)),
                        test_subjects,
                    )
                    break
            if selected_partition is not None:
                break

        if selected_partition is None:
            raise RuntimeError(
                "Could not find a feasible subject split. "
                f"Best train/test capacities: {best_capacities}. "
                "Increase search_trials or reduce the targets."
            )

        train_subjects, test_subjects = selected_partition
        train_candidates = defaultdict(list)
        test_candidates = defaultdict(list)
        for sample in metadata:
            destination = (
                train_candidates
                if sample["subject"] in train_subjects
                else test_candidates
            )
            destination[sample["label"]].append(sample["index"])

        sampling_rng = random.Random(seed)
        train_indices = []
        test_indices = []
        for label in labels:
            train_indices.extend(
                sampling_rng.sample(train_candidates[label], train_per_label)
            )
            test_indices.extend(
                sampling_rng.sample(test_candidates[label], test_per_label)
            )
        sampling_rng.shuffle(train_indices)
        sampling_rng.shuffle(test_indices)

        metadata_by_index = {
            sample["index"]: sample for sample in metadata
        }
        train_histogram = Counter(
            metadata_by_index[index]["label"] for index in train_indices
        )
        test_histogram = Counter(
            metadata_by_index[index]["label"] for index in test_indices
        )
        sampled_train_subjects = {
            metadata_by_index[index]["subject"] for index in train_indices
        }
        sampled_test_subjects = {
            metadata_by_index[index]["subject"] for index in test_indices
        }
        subject_overlap = sampled_train_subjects & sampled_test_subjects

        if set(train_histogram.values()) != {train_per_label}:
            raise AssertionError(f"Non-uniform train histogram: {train_histogram}")
        if set(test_histogram.values()) != {test_per_label}:
            raise AssertionError(f"Non-uniform test histogram: {test_histogram}")
        if set(train_histogram) != set(test_histogram):
            raise AssertionError("Train/test taxonomy labels do not match")
        if subject_overlap:
            raise AssertionError(
                f"Subject overlap found: {sorted(subject_overlap)}"
            )
        if set(train_indices) & set(test_indices):
            raise AssertionError("Train/test index overlap found")

        train_paths = [self.data_li[index] for index in train_indices]
        test_paths = [self.data_li[index] for index in test_indices]

        self.train_indices = train_paths
        self.test_indices = test_paths
        self.split_info = {
            "seed": seed,
            "labels": labels,
            "train_sample_count": len(train_paths),
            "test_sample_count": len(test_paths),
            "train_subjects": sorted(train_subjects),
            "test_subjects": sorted(test_subjects),
            "train_histogram": dict(sorted(train_histogram.items())),
            "test_histogram": dict(sorted(test_histogram.items())),
            "subject_overlap": sorted(subject_overlap),
            "validation_passed": True,
        }

        os.makedirs('cache', exist_ok=True)
        with open(train_cache_path, 'wb') as f:
            pickle.dump(train_paths, f)
        with open(test_cache_path, 'wb') as f:
            pickle.dump(test_paths, f)

        return train_paths, test_paths

if __name__ == '__main__':
    train_ds = HOGraspNetMANOContactDataset(split='train')
    (vertices, contact_map), label = train_ds[0]
    print(vertices.shape)
    print(contact_map.shape)
    # print(np.max(vertices), np.min(vertices))
    # print(np.max(contact_map), np.min(contact_map))

    '''
    train_ds = HOGraspNetMANOContactDataset(split='train')
    test_ds = HOGraspNetMANOContactDataset(split='test')

    print(len(train_ds), len(test_ds))

    taxo_cnt = [0 for i in range(0, 35)]

    for idx in range(len(train_ds)):
        taxo_label = int(train_ds.data_li[idx].split('/')[-1].split('.')[0].split('_')[-1])
        taxo_cnt[taxo_label] += 1

    for idx in range(0, len(taxo_cnt)):
        print(f"{idx}: {taxo_cnt[idx]}")
    '''