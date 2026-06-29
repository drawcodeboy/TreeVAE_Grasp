import torch
import random
import math

class RandomYawRotation:
    def __init__(self, angle_range=(-math.pi, math.pi), xyz_idx=(0, 1, 2)):
        self.angle_range = angle_range
        self.xyz_idx = xyz_idx

    def __call__(self, pc):
        # pc: (N, C)
        pc = pc.clone()

        theta = random.uniform(*self.angle_range)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        R = torch.tensor([
            [cos_t, -sin_t, 0.0],
            [sin_t,  cos_t, 0.0],
            [0.0,    0.0,   1.0],
        ], dtype=pc.dtype, device=pc.device)

        xyz = pc[:, self.xyz_idx]
        pc[:, self.xyz_idx] = xyz @ R.T

        return pc
    
class RandomJitter:
    def __init__(self, sigma=0.002, clip=0.005, xyz_idx=(0, 1, 2)):
        self.sigma = sigma
        self.clip = clip
        self.xyz_idx = xyz_idx

    def __call__(self, pc):
        pc = pc.clone()

        noise = torch.randn_like(pc[:, self.xyz_idx]) * self.sigma
        noise = torch.clamp(noise, -self.clip, self.clip)

        pc[:, self.xyz_idx] = pc[:, self.xyz_idx] + noise

        return pc