import os, sys
sys.path.append(os.getcwd())

import torch

'''
# Router n_ary 처리 확인
from models.networks import Router

model = Router(input_size=128, n_ary=2)

a = torch.randn(4, 128)

print(model(a).shape)

model = Router(input_size=128, n_ary=3)

a = torch.randn(4, 128)

print(model(a).shape)
'''