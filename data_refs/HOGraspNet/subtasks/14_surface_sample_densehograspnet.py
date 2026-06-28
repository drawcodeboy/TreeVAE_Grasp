import glob
import json
import os
import sys

import numpy as np
import torch
import trimesh as tm

# HOG_DIR이 환경 변수로 있으면 사용하고, 없으면 상대 경로로 복원합니다.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
HOG_DIR = os.environ.get('HOG_DIR', os.path.join(BASE_DIR, 'refs', 'HOGraspNet'))

# HOG 데이터 경로
LABELING_DATA_DIR = os.path.join(HOG_DIR, 'data', 'labeling_data')
OBJ_MODEL_DIR = os.path.join(HOG_DIR, 'data', 'obj_scanned_models')
MANO_MODEL_DIR = os.path.join(HOG_DIR, 'thirdparty', 'mano_v1_2', 'models')

# MANO import을 위해 HOG_DIR 및 thirdparty 경로 추가
sys.path.append(HOG_DIR)
sys.path.append(os.path.join(HOG_DIR, 'thirdparty', 'manopth'))
from thirdparty.manopth.manopth.manolayer import ManoLayer


def load_json(path):
    """UTF-8 BOM이 포함된 JSON 파일도 안전하게 읽습니다."""
    with open(path, 'r', encoding='utf-8-sig') as f:
        return json.load(f)


def find_object_mesh_path(object_id, object_file=None):
    """labeling_data JSON의 object 정보를 기반으로 obj mesh 파일을 찾습니다."""
    object_id = str(object_id)

    # 1) object_file이 명시된 경우, 정확히 일치하는 obj 파일 검색
    if object_file is not None:
        for root, _, files in os.walk(OBJ_MODEL_DIR):
            for fname in files:
                if fname == os.path.basename(object_file):
                    return os.path.join(root, fname)

    # 2) object_id로 시작하는 폴더 내부에서 .obj 파일 검색
    for entry in sorted(os.listdir(OBJ_MODEL_DIR)):
        if entry.startswith(f'{object_id}_'):
            entry_dir = os.path.join(OBJ_MODEL_DIR, entry)
            if os.path.isdir(entry_dir):
                for fname in sorted(os.listdir(entry_dir)):
                    if fname.lower().endswith('.obj'):
                        if object_file is None or os.path.basename(object_file) == fname:
                            return os.path.join(entry_dir, fname)

    # 3) 폴더 기준 검색 실패 시, 전체 경로 검색
    if object_file is not None:
        for root, _, files in os.walk(OBJ_MODEL_DIR):
            for fname in files:
                if fname.lower().endswith('.obj') and object_id in fname:
                    return os.path.join(root, fname)

    raise FileNotFoundError(
        f'Object mesh not found for object_id={object_id}, object_file={object_file}. '
        f'Search base: {OBJ_MODEL_DIR}'
    )


def load_trimesh(mesh_path):
    """trimesh로 mesh를 불러오고, scene일 경우 단일 mesh로 변환합니다."""
    mesh = tm.load(mesh_path, process=False)
    if isinstance(mesh, tm.Scene):
        mesh = tm.util.concatenate(mesh.dump())
    return mesh


def transform_object_mesh(mesh: tm.Trimesh, object_mat):
    """annotation의 object_mat를 이용해 object mesh를 world 좌표로 변환합니다."""
    verts = np.asarray(mesh.vertices)
    points_h = np.concatenate([verts, np.ones((len(verts), 1), dtype=np.float32)], axis=1)
    mat = np.asarray(object_mat, dtype=np.float32)
    transformed = (points_h @ mat.T)[:, :3]
    return tm.Trimesh(vertices=transformed, faces=np.asarray(mesh.faces), process=False)


def build_hand_mesh(mano_layer, mano_pose, mano_betas, mano_trans, mano_scale, mano_xyz_root, device='cpu'):
    """MANO 파라미터로 hand mesh를 만듭니다."""
    mano_pose = torch.FloatTensor(np.array(mano_pose)).to(device).reshape(1, -1)
    mano_betas = torch.FloatTensor(np.array(mano_betas)).to(device).reshape(1, -1)
    mano_trans = torch.FloatTensor(np.array(mano_trans)).to(device).reshape(1, -1)

    # shape: [1, 45], [1, 10], [1, 3]
    mano_param = torch.cat([mano_trans, mano_pose], dim=1)
    mano_verts, _ = mano_layer(mano_param, mano_betas)

    # HOG annotation에서 mano_scale / mano_xyz_root 적용
    # 참고: visualization.py와 동일한 후처리 로직입니다.
    mano_verts = (mano_verts / float(mano_scale)) + torch.FloatTensor(np.array(mano_xyz_root)).to(device)
    mano_verts = mano_verts[0].cpu().numpy()
    mano_faces = mano_layer.th_faces.cpu().numpy()
    return tm.Trimesh(vertices=mano_verts, faces=mano_faces, process=False)


def farthest_point_sampling(points: torch.Tensor, K: int):
    """Torch 기반 Farthest Point Sampling."""
    N = points.shape[0]
    if K <= 0 or N == 0:
        return points.new_empty((0, 3))
    if N <= K:
        return points

    selected_idx = torch.empty(K, dtype=torch.long, device=points.device)
    farthest = torch.randint(0, N, (1,), device=points.device).item()
    min_dist = torch.full((N,), float('inf'), device=points.device)

    for i in range(K):
        selected_idx[i] = farthest
        selected_point = points[farthest].unsqueeze(0)
        dist = torch.sum((points - selected_point) ** 2, dim=1)
        min_dist = torch.minimum(min_dist, dist)
        farthest = torch.argmax(min_dist).item()

    return points[selected_idx]


def sample_surface_pointcloud(mesh: tm.Trimesh, n_points: int, mul: int = 8, device='cpu'):
    """Mesh 표면에서 point cloud를 샘플링하고 FPS로 균일하게 만듭니다."""
    dense_points, _ = tm.sample.sample_surface(mesh, count=n_points * mul)
    dense_tensor = torch.from_numpy(dense_points).float().to(device)
    sampled = farthest_point_sampling(dense_tensor, K=n_points)
    return sampled.cpu().numpy()


def parse_label_file(label_path):
    """labeling_data JSON에서 hand/object mesh 정보와 camera/annotation 정보를 추출합니다."""
    anno = load_json(label_path)

    mesh_info = anno['Mesh'][0]
    object_id = anno['object']['id']
    object_file = mesh_info.get('object_file')
    object_mat = mesh_info['object_mat']
    mano_pose = mesh_info['mano_pose'][0]
    mano_betas = mesh_info['mano_betas'][0]
    mano_trans = mesh_info['mano_trans'][0]
    mano_scale = anno['hand']['mano_scale']
    mano_xyz_root = anno['hand']['mano_xyz_root']
    mano_side = mesh_info.get('mano_side', 'right')
    class_id = anno['annotations'][0]['class_id']
    class_name = anno['annotations'][0]['class_name']

    return {
        'label_path': label_path,
        'object_id': object_id,
        'object_file': object_file,
        'object_mat': object_mat,
        'mano_pose': mano_pose,
        'mano_betas': mano_betas,
        'mano_trans': mano_trans,
        'mano_scale': mano_scale,
        'mano_xyz_root': mano_xyz_root,
        'mano_side': mano_side,
        'class_id': class_id,
        'class_name': class_name,
        'annotations': anno
    }

def collect_label_paths(labeling_dir):
    """
    labeling_data 내부에서 각 Level 4 (subject) 디렉토리마다 
    랜덤으로 하나의 .json annotation 파일을 선택합니다.
    
    이렇게 하면 약 86,710개의 고유한 subject 폴더에서 각각 하나씩만 선택되어
    중복을 크게 줄일 수 있습니다.
    """
    # labeling_dir = "refs/HOGraspNet/data/labeling_data"
    import random
    
    label_paths = []
    subject_dirs_dict = {}  # 각 subject 폴더의 json 파일 목록
    
    # labeling_data 구조:
    # labeling_data/
    #   YYYYMM_Sxx_obj_YY_grasp_ZZ/trial_N/annotation/(mas|sub1|sub2|sub3)/
    
    for root, dirs, files in os.walk(labeling_dir):
        # annotation 폴더 내부의 subject 폴더 (mas, sub1, sub2, sub3)에 도달
        if 'annotation' in root:
            subject_dir = root  # 이미 annotation 폴더 내부
            
            # 각 subject 폴더 (mas, sub1, sub2, sub3)
            for subject_name in ['mas', 'sub1', 'sub2', 'sub3']:
                subject_path = os.path.join(root, subject_name)
                
                if os.path.isdir(subject_path):
                    json_files = [
                        os.path.join(subject_path, f) 
                        for f in os.listdir(subject_path) 
                        if f.endswith('.json')
                    ]
                    
                    #### 무엇을 저장하는가? ####
                    # 각 subject_experiment/trial/annotation/{mas,sub1,sub2,sub3} 폴더마다, 
                    # 해당 시퀀스의 json annotation 프레임 중 하나를 랜덤 선택한다.
                    if json_files:
                        selected_file = random.choice(json_files)
                        label_paths.append(selected_file)
    
    return sorted(label_paths)

def collect_label_paths_aux(labeling_dir, max_files=100):
    """labeling_dir 내부의 .json 파일 경로를 최대 max_files개만 수집합니다."""
    label_paths = []

    for root, dirs, files in os.walk(labeling_dir):
        for fname in files:
            if fname.endswith(".json"):
                label_paths.append(os.path.join(root, fname))

                if len(label_paths) >= max_files:
                    return label_paths

    return label_paths

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # MANO 모델 생성
    mano_layer = ManoLayer(
        side='right',
        mano_root=MANO_MODEL_DIR,
        use_pca=False,
        flat_hand_mean=True,
        center_idx=0,
        ncomps=45,
        root_rot_mode='axisang',
        joint_rot_mode='axisang'
    ).to(device)

    # label_files = collect_label_paths_aux(LABELING_DATA_DIR, 500)
    label_files = collect_label_paths(LABELING_DATA_DIR)
    print(f'found {len(label_files)} annotation files under {LABELING_DATA_DIR}')

    # 예시: 전체를 처리하려면 None, 일부만 테스트하려면 숫자 지정
    max_samples = None
    n_points = 1024
    output_dir = os.path.join(BASE_DIR, 'data', 'densehograspnet_pointcloud_p_1024')
    os.makedirs(output_dir, exist_ok=True)

    for idx, label_path in enumerate(label_files, start=1):
        if max_samples is not None and idx > max_samples:
            break

        info = parse_label_file(label_path)
        obj_mesh_path = find_object_mesh_path(info['object_id'], info['object_file'])

        object_mesh = load_trimesh(obj_mesh_path)
        object_mesh_world = transform_object_mesh(object_mesh, info['object_mat'])

        hand_mesh = build_hand_mesh(
            mano_layer,
            mano_pose=info['mano_pose'],
            mano_betas=info['mano_betas'],
            mano_trans=info['mano_trans'],
            mano_scale=info['mano_scale'],
            mano_xyz_root=info['mano_xyz_root'],
            device=device
        )

        object_pc = sample_surface_pointcloud(object_mesh_world, n_points=n_points, device=device)
        hand_pc = sample_surface_pointcloud(hand_mesh, n_points=n_points, device=device)

        os.makedirs(f"{output_dir}/{label_path.split('/')[-5]}", exist_ok=True)
        out_name = f"{label_path.split('/')[-5]}/{label_path.split('/')[-4]}__{label_path.split('/')[-2]}__{os.path.splitext(os.path.basename(label_path))[0]}__{info['class_id']}_{info['class_name']}"

        # 출력 예시: object/hand point cloud와 원본 라벨 정보 저장
        save_path = os.path.join(output_dir, f'{out_name}_pc.npz')
        np.savez_compressed(
            save_path,
            object_pc=object_pc,
            hand_pc=hand_pc,
            object_mesh_path=obj_mesh_path,
            label_path=label_path,
            object_id=info['object_id'],
            object_file=info['object_file'],
            class_id=info['class_id'],
            class_name=info['class_name'],
        )

        if idx % 50 == 0:
            print(f'processed {idx}/{len(label_files)} | saved {save_path}')

    print('done')

if __name__ == '__main__':
    main()