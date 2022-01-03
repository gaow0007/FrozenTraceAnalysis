import adaptdl 
import torch 
import adaptdl.torch
from adaptdl.torch.utils.misc import collect_atomic_layer_num 
from adaptdl.torch.layer_info import get_all_layer_info

import numpy as np 
import sys 
sys.path.insert(0, './') 
import torch.nn as nn
from model import MLMTask

for arch_info in [('WikiText2-bert', 'bert')]: 
    info_dict = dict()  
    model = MLMTask(
        28784, 768, 12,
        3072, 12, 0.2) 
    info_dict['layer_num'] = collect_atomic_layer_num(model) 
    flop_info, param_info, activation_info = get_all_layer_info(model,  [torch.arange(33).unsqueeze(0).long(), None])
    info_dict['param_info'] = param_info
    info_dict['flop_info'] = flop_info 
    info_dict['activation_info'] = activation_info 
    print(len(param_info), info_dict['layer_num']) 
    np.save('frozen/{}/{}.npy'.format(arch_info[0], 'app_info'), info_dict)
