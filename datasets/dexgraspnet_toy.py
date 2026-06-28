import os
import pickle

from torch.utils.data import Dataset
import torch
import numpy as np
from einops import rearrange
import random
import math

class DexGraspNetToyDataset(Dataset):
    def __init__(self,
                 cfg,
                 split='train',
                 transform=None):
        super().__init__()

        self.cfg = cfg

        self.data_li = []
        self.train_data_li, self.test_data_li = [], []
        self.transform = transform

        # NOTE: 이건 이전 setup
        # self._collect_toy_setup()
        # self._select_split()

        self._collect_toy_setup2()
        if split == 'train':
            self.data_li = self.train_data_li
        elif split == 'test':
            self.data_li = self.test_data_li
        else:
            raise Exception(f"Check split args, given {split}")

    def __len__(self):
        return len(self.data_li)
    
    def __getitem__(self, idx):
        data = np.load(os.path.join(self.cfg['data_dir'], self.data_li[idx]), allow_pickle=True).item()
        hand_pc = data['hand_pc'] # (1, n_points, 3)
        obj_pc = data['object_pc'] # (n_points, 3)
        label = self.data_li[idx][:-13]

        hand_pc = np.squeeze(hand_pc) # (n_points, 3)

        total_pc = np.concatenate([hand_pc, obj_pc], axis=0) # (n_points*2, 3)
        total_pc = torch.tensor(total_pc, dtype=torch.float32)

        print(total_pc.shape) # (n_points*2, 3)
        

        if self.transform is not None:
            total_pc = self.transform(total_pc)

        total_pc = rearrange(total_pc, 'n c -> c n') # (3, n_points*2)

        return total_pc

    def _collect_toy_setup(self):
        os.makedirs('cache', exist_ok=True)
        if not os.path.exists(os.path.join("cache", "toy_dexgraspnet.pkl")):
            samples_li = os.listdir(self.cfg['data_dir'])
            sample_scaled_object_name = [sample_name[:-13] for sample_name in samples_li]

            # (1) Object를 랜덤으로 뽑기 (단, scale이 다르지만 동일한 object인 경우, 해당 object는 포함하지 않음)
            n_objs = self.cfg['n_objs']
            cnt_objs = 0
            selected_objects = []

            def _scale_check(sample_name):
                # scale은 다르지만, 같은 object일 경우 toy dataset에 넣지 않으려고 함.
                if sample_name[:-4] not in [selected_obj[:-4] for selected_obj in selected_objects]:
                    return True
                else:
                    return False

            def _num_grasps_check(sample_name):
                return len([x for x in samples_li if x.startswith(sample_name)]) >= self.cfg['grasps_per_obj']

            while (cnt_objs < n_objs):
                sample_name = random.choice(sample_scaled_object_name)
                
                if len(selected_objects) == 0 or (_scale_check(sample_name) and _num_grasps_check(sample_name)):
                    selected_objects.append(sample_name)
                    cnt_objs += 1

            # (2) 선택된 object들에 대하여 n개의 grasp 데이터를 가져오기
            grasps_per_obj = self.cfg['grasps_per_obj']

            for obj in selected_objects:
                matched = [x for x in samples_li if x.startswith(obj)]
                random.shuffle(matched)
                matched = matched[:grasps_per_obj]
                self.data_li.extend(matched)

            with open(os.path.join("cache", "toy_dexgraspnet.pkl"), "wb") as f:
                pickle.dump(self.data_li, f)
        else:
            with open(os.path.join("cache", "toy_dexgraspnet.pkl"), "rb") as f:
                self.data_li = pickle.load(f)
    
    def _collect_toy_setup2(self):
        os.makedirs('cache', exist_ok=True)
        if (not os.path.exists(os.path.join("cache", "train_setup2_toy_dexgraspnet.pkl"))) and (not os.path.exists(os.path.join("cache", "test_setup2_toy_dexgraspnet.pkl"))):
            samples_li = os.listdir(self.cfg['data_dir'])
            sample_scaled_object_name = [sample_name[:-13] for sample_name in samples_li]

            # (1) Object를 랜덤으로 뽑기 (단, scale이 다르지만 동일한 object인 경우, 해당 object는 포함하지 않음)
            n_objs = self.cfg['n_objs']
            cnt_objs = 0
            selected_objects = []

            def _scale_check(sample_name):
                # scale은 다르지만, 같은 object일 경우 toy dataset에 넣지 않으려고 함.
                if sample_name[:-4] not in [selected_obj[:-4] for selected_obj in selected_objects]:
                    return True
                else:
                    return False

            while (cnt_objs < n_objs):
                sample_name = random.choice(sample_scaled_object_name)
                
                if len(selected_objects) == 0 or _scale_check(sample_name):
                    selected_objects.append(sample_name)
                    cnt_objs += 1

            # (2) 선택된 object들에 대하여 모든 grasp 데이터를 가져오기
            for obj in selected_objects:
                matched = [x for x in samples_li if x.startswith(obj)]
                random.shuffle(matched)
                matched = matched[:self.cfg['grasps_per_obj']]
                self.train_data_li.extend(matched[:int(0.7 * len(matched))])
                self.test_data_li.extend(matched[int(0.7 * len(matched)):]) 

            with open(os.path.join("cache", "train_setup2_toy_dexgraspnet.pkl"), "wb") as f:
                pickle.dump(self.train_data_li, f)
            with open(os.path.join("cache", "test_setup2_toy_dexgraspnet.pkl"), "wb") as f:
                pickle.dump(self.test_data_li, f)
        else:
            with open(os.path.join("cache", "train_setup2_toy_dexgraspnet.pkl"), "rb") as f:
                self.train_data_li = pickle.load(f)
            with open(os.path.join("cache", "test_setup2_toy_dexgraspnet.pkl"), "rb") as f:
                self.test_data_li = pickle.load(f)

    def _select_split(self):
        '''
        리스트 내에서 object 별 grasp끼리 묶여 있다는 전제 하에 해당 method를 사용할 수 있으므로 유념할 것!
        '''

        # 아래 ratio에 따르면서 split 하려면 object 종류 수가 10배수가 되어야 함
        train_ratio, val_ratio, test_ratio = 0.7, 0.1, 0.2
        train_num = int(train_ratio * self.cfg['n_objs'] * self.cfg['grasps_per_obj'])
        val_num = int(val_ratio * self.cfg['n_objs'] * self.cfg['grasps_per_obj'])
        test_num = int(test_ratio * self.cfg['n_objs'] * self.cfg['grasps_per_obj'])

        if self.cfg['split'] == 'train':
            self.data_li = self.data_li[:train_num]
        elif self.cfg['split'] == 'val':
            self.data_li = self.data_li[train_num:train_num+val_num]
        elif self.cfg['split'] == 'test':
            self.data_li = self.data_li[train_num+val_num:train_num+val_num+test_num]
        else:
            raise Exception(f"Check your cfg['split'], {self.cfg['split']}")