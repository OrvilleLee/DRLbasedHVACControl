# import torch
#
# flag = torch.cuda.is_available()
# print(flag)
#
# ngpu = 1
# device = torch.device('cuda:0' if (flag and ngpu>0) else 'cpu')
# print(device)
# print(torch.cuda.get_device_name(0))
# print(torch.rand(3,3).cuda())
#
# cuda_version = torch.version.cuda
# print(f'CUDA version: {cuda_version}')
#
# cudnn_version = torch.backends.cudnn.version()
# print(f'cudnn version: {cudnn_version}')
#
from pprint import pprint
from collections import deque
import random
from tools import HVAC_action_map

print(random.choice(HVAC_action_map()))