import adaptdl 
import torch 
import adaptdl.torch
from adaptdl.torch.utils.misc import collect_atomic_layer_num 
from adaptdl.torch.layer_info import get_all_layer_info

import numpy as np 
import sys 
sys.path.insert(0, './') 
from model.yolov3 import Yolov3

for arch_info in [('VOC-yolo', 'yolo')]: 
    info_dict = dict() 
    # model = VGG('VGG19') if arch_info[1] == 'VGG19' else eval(arch_info[1])() 
    model = Yolov3().cuda()
    info_dict['layer_num'] = collect_atomic_layer_num(model)  
    flop_info, param_info, activation_info = get_all_layer_info(model, torch.randn(1, 3, 416, 416)) 
    info_dict['param_info'] = param_info
    info_dict['flop_info'] = flop_info  
    info_dict['activation_info'] = activation_info 
    print(len(param_info), info_dict['layer_num']) # 117
    np.save('frozen/{}/{}.npy'.format(arch_info[0], 'app_info'), info_dict)
