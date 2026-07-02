"""Five-layer fully-connected autoencoder blocks for 3D hand joints."""
import torch
from torch import nn


def _make_mlp(input_dim, output_dim, hidden_dims, dropout, final_relu):
    """Create five Linear layers with ReLU/dropout between stacks."""
    hidden_dims = tuple(hidden_dims)
    if len(hidden_dims) != 4:
        raise ValueError("hidden_dims must have four values for five Linear layers")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    widths = (input_dim, *hidden_dims, output_dim)
    layers = []
    for index, (in_dim, out_dim) in enumerate(zip(widths[:-1], widths[1:])):
        is_last = index == 4
        layers.append(nn.Linear(in_dim, out_dim))
        if not is_last or final_relu:
            layers.append(nn.ReLU(inplace=True))
        if not is_last and dropout:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class HandJointEncoder(nn.Module):
    """Encode joints shaped (B, 3, J), (B, J, 3), or (B, 3*J)."""
    def __init__(self, encoded_size, num_joints=21,
                 hidden_dims=(256, 256, 128, 128), dropout=0.2):
        super().__init__()
        self.num_joints = num_joints
        self.input_dim = 3 * num_joints
        self.layers = _make_mlp(
            self.input_dim, encoded_size, hidden_dims, dropout, final_relu=True
        )

    def forward(self, joints: torch.Tensor):
        if joints.ndim == 3:
            if joints.shape[1:] == (self.num_joints, 3):
                joints = joints.transpose(1, 2)
            elif joints.shape[1:] != (3, self.num_joints):
                raise ValueError(f"Unexpected joint shape: {tuple(joints.shape)}")
            joints = joints.reshape(joints.shape[0], self.input_dim)
        elif joints.ndim != 2 or joints.shape[1] != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} coordinates per sample")
        return self.layers(joints), None, None


class HandJointDecoder(nn.Module):
    """Decode latent vectors into root-relative joints shaped (B, 3, J)."""
    def __init__(self, input_shape, num_joints=21,
                 hidden_dims=(128, 128, 256, 256), dropout=0.2):
        super().__init__()
        self.num_joints = num_joints
        # No final ReLU: root-relative coordinates may be negative.
        self.layers = _make_mlp(
            input_shape, 3 * num_joints, hidden_dims, dropout, final_relu=False
        )

    def forward(self, latent: torch.Tensor):
        return self.layers(latent).reshape(latent.shape[0], 3, self.num_joints)


def get_encoder_skeleton(architecture, encoded_size, num_joints=21,
                         hidden_dims=(256, 256, 128, 128), dropout=0.2):
    if architecture not in {"mlp", "fc"}:
        raise ValueError(f"Unknown skeleton encoder: {architecture!r}")
    return HandJointEncoder(encoded_size, num_joints, hidden_dims, dropout)


def get_decoder_skeleton(architecture, input_shape, num_joints=21,
                         hidden_dims=(128, 128, 256, 256), dropout=0.2):
    if architecture not in {"mlp", "fc"}:
        raise ValueError(f"Unknown skeleton decoder: {architecture!r}")
    return HandJointDecoder(input_shape, num_joints, hidden_dims, dropout)
