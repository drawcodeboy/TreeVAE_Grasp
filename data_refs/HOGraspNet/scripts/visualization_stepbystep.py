# mesh
# contact map 결합해서
# html로 시각화
import argparse
import torch
import os, sys
sys.path.append(os.environ['HOG_DIR'])
sys.path.append(os.path.join(os.environ['HOG_DIR'], "thirdparty/manopth"))

from torch.utils.data import DataLoader
import plotly.graph_objects as go
import numpy as np

from HOG_mesh_dataloader import HOGMeshDataset
from config import cfg
from thirdparty.manopth.manopth.manolayer import ManoLayer

def save_mano_html(verts, faces, contact_map=None, save_path="mano_mesh.html"):
    """
    verts: torch.Tensor or np.ndarray, shape (778, 3)
    faces: torch.Tensor or np.ndarray, shape (F, 3)
    contact_map: torch.Tensor or np.ndarray, shape (778,)
    """
    if torch.is_tensor(verts):
        verts = verts.detach().cpu().numpy()
    if torch.is_tensor(faces):
        faces = faces.detach().cpu().numpy()

    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=i,
                j=j,
                k=k,
                opacity=1.0,
                intensity=contact_map,
                # intensitymode='cell',
                colorbar=dict(
                    title="Contact",
                ),
            )
        ]
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        title="MANO Hand Mesh",
    )

    fig.write_html(save_path)
    print(f"saved: {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_path",
        type=str,
        default="./data"
    )

    args = parser.parse_args()

    base_path = args.base_path

    ds = HOGMeshDataset(setup='s0', split='train', db_path=base_path)
    dl = DataLoader(ds, batch_size=1, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # MANO Layer
    # MANO parameter 넣어서 Hand mesh 주는 파트
    # Input: MANO parameters -> 검증 필요
    # Output: Hand mesh (B, 778, 3), Hand joints (B, 21, 3)
    mano_layer = ManoLayer(center_idx=None, # 어떤 joint를 중심으로 할 것인가, None이면 그대로
                           flat_hand_mean=True, # 기준 pose를 완전히 펴진 손으로 할 것인가, False로 두어서 평균적인 pose로 할 것인가
                           ncomps=45, # PCA component 수, 45로 하면 거의 원본과 동일한 손 mesh가 나옴 -> pose parameter 개수라 생각하면 될 듯
                                      # 45보다 작게 하더라도 모델 내에서 이를 45로 복원, 그럼 PCA를 왜 쓰냐? 이건 공부해봐야 할 듯
                           side='right', # 각 손에 따른 모델, 서로 PCA basis가 다르기 때문에 따로 모델이 존재, 그래서 마음대로 쓰는 게 아님
                           mano_root=os.path.join(os.environ['HOG_DIR'], "thirdparty/mano_v1_2/models"), # MANO model 경로
                           use_pca=False, # 입력을 PCA 계수로 받을 거냐, 아니면 full pose로 받을 거냐
                           root_rot_mode="axisang", # 손이 공간에서 어떻게 돌아가 있냐
                           joint_rot_mode="axisang", # 각 관절의 rotation을 어떤 형태로 표현할지
                           robust_rot=False) # 6D rotation -> matrix 변환을 더 안정적으로 할지
    
    # hand mesh template
    # MANO 손 모델에 미리 정의된 face 정보
    # 고정된 mesh 연결 템플릿, 여기 face 얼굴 아님 면으로 이해해야 함
    hand_faces_template = mano_layer.th_faces.repeat(1, 1, 1) # .repeat()는 각 dimension 별로 복제한 거라 그대로임
    
    # Renderer는 우선 구현하지 않겠음 (visualization_meshonly.py 참고)
    # HTML로 나중에 시각화할 거라서

    for i in range(30):
        sample = ds[i]

        # MANO parameter -> hand mesh
        anno = sample['anno_data']

        hand_mano_rot = anno['Mesh'][0]['mano_trans']
        hand_mano_pose = anno['Mesh'][0]['mano_pose']
        hand_mano_shape = anno['Mesh'][0]['mano_betas']
        hand_contact_map = anno['contact']

        hand_mano_rot = torch.FloatTensor(hand_mano_rot).to(device)
        hand_mano_pose = torch.FloatTensor(hand_mano_pose).to(device)
        hand_mano_shape = torch.FloatTensor(hand_mano_shape).to(device)
        hand_contact_map = torch.FloatTensor(hand_contact_map).to(device)

        mano_param = torch.cat([hand_mano_rot, hand_mano_pose], dim=1).to(device)
        mano_verts, mano_joints = mano_layer(mano_param, hand_mano_shape)
        contact_map = torch.squeeze(hand_contact_map)

        # MANO는 vertex랑 joint 둘 다 준다.
        # print(mano_verts.shape) # (1, 778, 3)
        # print(mano_joints.shape) # (1, 21, 3)
        # print(contact_map.shape) # (778,)

        # Rendering -> HTML visualization
        os.makedirs("./vis", exist_ok=True)
        print(sample['label_path'])
        
        label_path = sample['label_path']
        seq_name = label_path.split('/')[-5]
        trial_name = label_path.split('/')[-4]
        misc_name = label_path.split('/')[-1].split('.')[0]

        object_name = anno['Mesh'][0]['object_name']
        grasp_name = anno['Mesh'][0]['class_name']

        save_mano_html(mano_verts[0], hand_faces_template[0], contact_map=contact_map, save_path=f"./vis/mesh_{object_name}__{grasp_name}__{seq_name}_{trial_name}_{misc_name}.html")
        print("Visualization saved in html")

if __name__ == '__main__':
    main()