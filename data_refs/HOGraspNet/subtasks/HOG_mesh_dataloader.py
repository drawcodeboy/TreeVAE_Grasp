"""HOGraspNet MANO mesh dataset."""

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import json
import torch
import numpy as np
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
from tqdm import tqdm
import pickle
from config import cfg
from util.utils import extractBbox

class HOGMeshDataset():
    def __init__(self, setup, split, db_path, use_aug=False, load_pkl=True, path_pkl=None):
        """Constructor for MANO mesh dataset.
        Args:
        setup: Setup name. 'travel_all', 's0', 's1', 's2', 's3', or 's4'
        split: Split name. 'train', 'val', or 'test'
        db_path: path to dataset folder.
        use_aug: Use crop&augmented rgb data if exists.
        load_pkl: Use saved pkl if exists.
        """

        self._setup = setup
        self._split = split
        self._use_aug = use_aug
        self._mano_only = True  # Always load MANO data only

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._base_dir = db_path
        self._base_anno = os.path.join(self._base_dir, 'labeling_data')
        self._base_source = os.path.join(self._base_dir, 'source_data')
        self._base_source_aug = os.path.join(self._base_dir, 'source_augmented')

        self._base_extra = os.path.join(self._base_dir, 'extra_data')
        self._obj_model_dir = os.path.join(self._base_dir, 'obj_scanned_models')

        self._h = 480
        self._w = 640

        self.camIDset = cfg._CAMIDSET

        # create pkl once, load if exist.
        if path_pkl == None:
            self._data_pkl_pth = f'cfg/{setup}_{split}_mesh.pkl'
        else:
            self._data_pkl_pth = path_pkl

        ## CHECK DATA
        assert os.path.isdir(self._base_anno), "labeling data is not set, we require at least annotation data to run dataloader"

        ## MINING SEQUENCE INFOS
        self._SUBJECTS, self._OBJ_IDX, self._GRASP_IDX, self._OBJ_GRASP_PAIR = [], [], [], []

        seq_list = os.listdir(self._base_anno)
        self._seq_dict_list = []
        for idx, seq in enumerate(seq_list) :
            seq_info = {}
            seq_split = seq.split('_')

            seq_info['idx']  = idx
            seq_info['seqName'] = seq
            # seq_info['date'] = seq_split[0]
            seq_info['subject'] = seq_split[1]
            seq_info['obj_idx'] = seq_split[3]
            seq_info['grasp_idx'] = seq_split[5]
            seq_info['obj_grasp_pair'] = [seq_split[3],seq_split[5]]

            if seq_info['subject'] not in self._SUBJECTS :
                self._SUBJECTS.append(seq_info['subject'])

            if seq_info['obj_idx'] not in self._OBJ_IDX :
                self._OBJ_IDX.append(seq_info['obj_idx'])

            if seq_info['grasp_idx'] not in self._GRASP_IDX :
                self._GRASP_IDX.append(seq_info['grasp_idx'])

            if seq_info['obj_grasp_pair'] not in self._OBJ_GRASP_PAIR :
                self._OBJ_GRASP_PAIR.append(seq_info['obj_grasp_pair'])

            self._seq_dict_list.append(seq_info)


        ## TRAIN / TEST / VALID SPLIT

        # ALL
        if self._setup == 'travel_all':
            subject_ind = self._SUBJECTS
            serial_ind = self.camIDset
            obj_grasp_pair_ind = self._OBJ_GRASP_PAIR
            trial_ind = 'full'      # 'full', 'train', 'val', 'test'

        # s0 : UNSEEN TRIAL
        if self._setup == 's0':
            subject_ind = self._SUBJECTS
            serial_ind = self.camIDset
            obj_grasp_pair_ind = self._OBJ_GRASP_PAIR

            trial_ind = self._split     # 'full', 'train', 'val', 'test'

        # s1 : UNSEEN SUBJECTS
        if self._setup == 's1':
            serial_ind = self.camIDset
            obj_grasp_pair_ind = self._OBJ_GRASP_PAIR
            trial_ind = 'full'      # 'full', 'train', 'val', 'test'

            if self._split == 'train':
                subject_ind = self._SUBJECTS[:73]
            if self._split == 'test':
                subject_ind = self._SUBJECTS[73:]
            if self._split == 'val':
                subject_ind = self._SUBJECTS[:10]

        # s2 : UNSEEN CAM
        if self._setup == 's2':
            subject_ind = self._SUBJECTS
            obj_grasp_pair_ind = self._OBJ_GRASP_PAIR
            trial_ind = 'full'      # 'full', 'train', 'val', 'test'

            if self._split == 'train':
                serial_ind = self.camIDset[:-1]
            if self._split == 'test':
                serial_ind = self.camIDset[-1]
            if self._split == 'val':
                serial_ind = self.camIDset[0]

        # s3 : UNSEEN OBJECTS
        if self._setup == 's3':
            subject_ind = self._SUBJECTS
            serial_ind = self.camIDset
            trial_ind = 'full'      # 'full', 'train', 'val', 'test'

            train_pair, test_pair = [], []
            for pair in self._OBJ_GRASP_PAIR :
                if pair[0] in cfg._TEST_OBJ_LIST :
                    test_pair.append(pair)
                else :
                    train_pair.append(pair)

            if self._split == 'train':
                obj_grasp_pair_ind = train_pair
            if self._split == 'test':
                obj_grasp_pair_ind = test_pair
            if self._split == 'valid':
                obj_grasp_pair_ind = train_pair[:int(len(train_pair)/5)]


        # s4 : UNSEEN GRASP TAXONOMY
        if self._setup == 's4':
            subject_ind = self._SUBJECTS
            serial_ind = self.camIDset
            trial_ind = 'full'      # 'full', 'train', 'val', 'test'

            train_pair, test_pair = [], []
            for pair in self._OBJ_GRASP_PAIR :
                if pair[1] in cfg._TEST_GRASP_LIST :
                    test_pair.append(pair)
                else :
                    train_pair.append(pair)

            if self._split == 'train':
                obj_grasp_pair_ind = train_pair
            if self._split == 'test':
                obj_grasp_pair_ind = test_pair
            if self._split == 'valid':
                obj_grasp_pair_ind = train_pair[:int(len(train_pair)/5)]


        #########################################


        # for each object has its mapping index which contains s,t,c,f (subject,trial,cam,frame)
        total_count = 0
        self.load = False
        self.mapping = [] # its location
        self.cam_param_dict = {}

        sample_seq_dict = {}

        ## load pkl if exist
        if os.path.isfile(self._data_pkl_pth) and load_pkl:
            print(f"loading from saved pkl {self._data_pkl_pth}")
            with open(self._data_pkl_pth, 'rb') as handle:
                dict_data = pickle.load(handle)

            self.dataset_samples = dict_data['data']
            self.mapping = dict_data['mapping']
            self.cam_param_dict = dict_data['camera_info']
        else:
            for seqIdx, seq in enumerate(tqdm(self._seq_dict_list)):
                # skip if not target sequence
                if seq['subject'] not in subject_ind :
                    continue
                if seq['obj_grasp_pair'] not in obj_grasp_pair_ind :
                    continue

                sample_trial_dict = {}
                cam_param_dict_trial = {}

                seqName = seq['seqName']
                seqDir = os.path.join(self._base_anno, seqName)
                for trialIdx, trialName in enumerate(sorted(os.listdir(seqDir))):
                    # skip if not target trial
                    if trial_ind == 'train' and trialIdx == 0:
                        continue
                    if trial_ind == 'test' and trialIdx != 0:
                        continue
                    if trial_ind == 'valid' and trialIdx != 1:
                        continue

                    anno_base_path = os.path.join(seqDir, trialName, 'annotation')
                    valid_cams = os.listdir(anno_base_path)

                    Ks_dict = {}
                    Ms_dict = {}

                    for camID in self.camIDset:
                        if camID in valid_cams:
                            anno_list = os.listdir(os.path.join(anno_base_path, camID))
                            anno_path = os.path.join(anno_base_path, camID, anno_list[0])

                            with open(anno_path, 'r', encoding='UTF-8 SIG') as file:
                                anno = json.load(file)

                            Ks = torch.FloatTensor(np.squeeze(np.asarray(anno['calibration']['intrinsic']))).to(self.device)
                            Ms = np.squeeze(np.asarray(anno['calibration']['extrinsic']))
                            Ms = np.reshape(Ms, (3, 4))
                            # Ms[:, -1] = Ms[:, -1] / 10.0
                            Ms = torch.Tensor(Ms).to(self.device)

                            Ks_dict[camID] = Ks
                            Ms_dict[camID] = Ms

                        else:
                            Ks_dict[camID] = None
                            Ms_dict[camID] = None

                    self.valid_cams = valid_cams

                    cam_param_dict_trial[trialName] = {}
                    cam_param_dict_trial[trialName]['Ks'] = Ks_dict
                    cam_param_dict_trial[trialName]['Ms'] = Ms_dict

                    self.anno_dict, rgb_dict, depth_dict, flag_crop = self.load_data(seqName, trialName, valid_cams)

                    sample_cam_dict = {}
                    for camIDX, camID in enumerate(valid_cams):
                        if camID not in serial_ind:
                            continue

                        sample_idx_dict = {}
                        for anno_idx, anno_path in enumerate(self.anno_dict[camID]) :
                            sample = {
                                'label_path': self.anno_dict[camID][anno_idx],
                                'obj_ids': seq['obj_idx'],
                                'taxonomy': seq['grasp_idx'],
                                'flag_crop': flag_crop
                            }

                            frame_num = anno_path.split('/')[-1].split('_')[-1][:-5]
                            self.mapping.append([seqName,trialName,camID,str(anno_idx),str(frame_num)])

                            sample_idx_dict[str(anno_idx)] = sample
                        sample_cam_dict[camID] = sample_idx_dict
                    sample_trial_dict[trialName] = sample_cam_dict
                sample_seq_dict[seqName] = sample_trial_dict

                self.cam_param_dict[seqName] = cam_param_dict_trial

            self.dataset_samples = sample_seq_dict

            ## save pkl
            dict_data = {}
            dict_data['data'] = self.dataset_samples
            dict_data['mapping'] = self.mapping
            dict_data['camera_info'] = self.cam_param_dict

            os.makedirs("cfg", exist_ok=True)
            with open(self._data_pkl_pth, 'wb') as handle:
                pickle.dump(dict_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


        assert len(self.mapping) > 0, "downloaded data is not enough for given split"

    def get_mapping(self):
        return self.mapping

    def load_data(self, seqName, trialName, valid_cams):

        anno_base_path = os.path.join(self._base_anno, seqName, trialName, 'annotation')
        rgb_base_path = os.path.join(self._base_source, seqName, trialName, 'rgb')
        depth_base_path = os.path.join(self._base_source, seqName, trialName, 'depth')

        # use cropped image if exists.
        flag_crop = False
        rgb_aug_base_path = os.path.join(self._base_source_aug, seqName, trialName)
        if os.path.isdir(rgb_aug_base_path):
            flag_crop = True
            depth_base_path = os.path.join(rgb_aug_base_path, 'depth_crop')
            if self._use_aug:
                rgb_base_path = os.path.join(rgb_aug_base_path, 'rgb_aug')
            else:
                rgb_base_path = os.path.join(rgb_aug_base_path, 'rgb_crop')


        anno_dict = {}
        rgb_dict = {}
        depth_dict = {}

        for camIdx, camID in enumerate(self.camIDset):
            anno_dict[camID] = []
            rgb_dict[camID] = []
            depth_dict[camID] = []

            if camID in valid_cams:
                anno_list = os.listdir(os.path.join(anno_base_path, camID))

                for anno in anno_list:
                    anno_path = os.path.join(anno_base_path, camID, anno)
                    anno_dict[camID].append(anno_path)

        return anno_dict, rgb_dict, depth_dict, flag_crop

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        # s : subject , t : trial,  c : camera, i : idx, f : framenum
        s, t, c, i, f = self.mapping[idx]

        sample = self.dataset_samples[s][t][c][i]

        ####### load data and set sample #####
        label_path = sample['label_path']
        with open(label_path, 'r', encoding='UTF-8 SIG') as file:
            anno_data = json.load(file)

        # Extract MANO data
        mano_data = {
            'mano_scale': anno_data['hand']['mano_scale'],
            'mano_xyz_root': np.array(anno_data['hand']['mano_xyz_root']),
            'hand_3d_joints': np.array(anno_data['hand']['3D_pose_per_cam']),
            'hand_2d_joints': np.array(anno_data['hand']['projected_2D_pose_per_cam'])
        }
        
        # Extract true MANO parameters from Mesh section
        if 'Mesh' in anno_data and len(anno_data['Mesh']) > 0:
            mesh_info = anno_data['Mesh'][0]
            mano_data.update({
                'mano_pose': np.array(mesh_info['mano_pose'][0]),  # [45]
                'mano_betas': np.array(mesh_info['mano_betas'][0]), # [10] 
                'mano_trans': np.array(mesh_info['mano_trans'][0]), # [3]
                'mano_side': mesh_info.get('mano_side', 'right'),
                'object_mat': np.array(mesh_info['object_mat']),
                'object_file': mesh_info.get('object_file', None)
            })

        hand_2d = np.squeeze(np.asarray(anno_data['hand']['projected_2D_pose_per_cam']))
        bbox, _ = extractBbox(hand_2d)

        sample['anno_data'] = anno_data
        sample['mano_data'] = mano_data
        sample['bbox'] = bbox
        sample['contact'] = np.array(anno_data['contact'])

        sample['camera']  = c
        sample['intrinsics'] = self.cam_param_dict[s][t]['Ks'][c]
        sample['extrinsics'] = self.cam_param_dict[s][t]['Ms'][c]

        return sample


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--setup', type=str, default='s2', help='Setup name')
    parser.add_argument('--split', type=str, default='test', help='Split name')
    parser.add_argument('--base_path', type=str, default='./data', help='Base path to dataset')
    args = parser.parse_args()

    print("loading MANO mesh dataset ... ", args.setup + '_' + args.split)

    db_path = args.base_path
    HOG = HOGMeshDataset(args.setup, args.split, db_path)

    print("db len: ", len(HOG))
    data = HOG[0]
    print("Sample keys:", list(data.keys()))
    if 'mano_data' in data:
        print("MANO data keys:", list(data['mano_data'].keys()))
        print("MANO scale:", data['mano_data']['mano_scale'])
        print("MANO root:", data['mano_data']['mano_xyz_root'])
        print("Hand 3D joints shape:", data['mano_data']['hand_3d_joints'].shape)
        print("Hand 2D joints shape:", data['mano_data']['hand_2d_joints'].shape)
        if 'mano_pose' in data['mano_data']:
            print("MANO pose shape:", data['mano_data']['mano_pose'].shape)
            print("MANO betas shape:", data['mano_data']['mano_betas'].shape)
            print("MANO trans shape:", data['mano_data']['mano_trans'].shape)
            print("MANO side:", data['mano_data']['mano_side'])
            print("Object file:", data['mano_data']['object_file'])
    print("Sample keys:", list(data.keys()))


if __name__ == '__main__':
    main()