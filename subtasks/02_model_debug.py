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

from models.model import TreeVAE
from utils.utils import prepare_config
from pathlib import Path
import yaml

config_path = Path("configs/hograspnet_uniform_toy.yml")

with config_path.open(mode='r') as yamlfile:
    configs = yaml.safe_load(yamlfile)

model = TreeVAE(**configs['training'])

x = torch.randn((16, 3, 4096))
return_dict = model(x)

print(return_dict)