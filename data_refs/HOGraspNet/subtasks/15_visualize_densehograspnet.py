import argparse
import glob
import os
import random

import numpy as np
import plotly.graph_objects as go


def save_point_cloud_html(object_points, hand_points, save_path,
                          object_name="object_pc", hand_name="hand_pc",
                          object_color="green", hand_color="blue",
                          object_size=2, hand_size=2):
    if object_points.ndim == 3:
        object_points = object_points[0]
    if hand_points.ndim == 3:
        hand_points = hand_points[0]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=object_points[:, 0],
                y=object_points[:, 1],
                z=object_points[:, 2],
                mode="markers",
                name=object_name,
                marker=dict(size=object_size, color=object_color, opacity=0.8)
            ),
            go.Scatter3d(
                x=hand_points[:, 0],
                y=hand_points[:, 1],
                z=hand_points[:, 2],
                mode="markers",
                name=hand_name,
                marker=dict(size=hand_size, color=hand_color, opacity=0.8)
            )
        ]
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data"
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )
    fig.write_html(save_path)


def load_random_npz(data_dir):
    file_paths = sorted(glob.glob(os.path.join(data_dir, random.choice(os.listdir(data_dir)), '*.npz')))
    if len(file_paths) == 0:
        raise FileNotFoundError(f'No .npz files found in {data_dir}')
    return random.choice(file_paths)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize a random dense HOG grasp point cloud as HTML.'
    )
    parser.add_argument(
        '--data-dir',
        default='data/densehograspnet_pointcloud',
        help='Directory containing .npz point cloud files.'
    )
    parser.add_argument(
        '--output',
        default='vis/densehograspnet2_pointcloud.html',
        help='HTML output path.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducible selection.'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    for i in range(20):
        npz_path = load_random_npz(args.data_dir)
        data = np.load(npz_path, allow_pickle=True)

        if 'hand_pc' not in data or 'object_pc' not in data:
            raise KeyError('Expected keys hand_pc and object_pc in the .npz file')

        object_pc = data['object_pc']
        hand_pc = data['hand_pc']
        print(data['class_id'], data['class_name'])

        for key in data.keys():
            print(f'{key}: shape={data[key].shape}, dtype={data[key].dtype}')

        output = f"vis/densehograspnet2_pointcloud_{i:02d}.html"

        print(f'Visualizing: {npz_path}')
        save_point_cloud_html(
            object_points=object_pc,
            hand_points=hand_pc,
            save_path=output,
            object_name='object_pc',
            hand_name='hand_pc',
            object_color='green',
            hand_color='blue',
            object_size=2,
            hand_size=2
        )
        print(f'Saved HTML visualization to: {output}_{i:02d}')


if __name__ == '__main__':
    main()
