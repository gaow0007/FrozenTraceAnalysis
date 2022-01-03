import adaptdl 
import adaptdl.torch
from adaptdl.torch.utils.misc import collect_atomic_layer_num 
from adaptdl.torch.layer_info import get_all_layer_info

import numpy as np 
from models import * 


for arch_info in [('cifar10-ResNet18', 'ResNet18'), ('cifar10-VGG19', 'VGG19'), ('cifar10-ResNet50', 'ResNet50'), ('cifar10-MobileNetV2', 'MobileNetV2'), ('cifar10-GoogLeNet', 'GoogLeNet')]: 
    info_dict = dict() 
    model = VGG('VGG19') if arch_info[1] == 'VGG19' else eval(arch_info[1])() 
    info_dict['layer_num'] = collect_atomic_layer_num(model) 
    flop_info, param_info, activation_info = get_all_layer_info(model, torch.randn(1, 3, 32, 32))
    info_dict['param_info'] = param_info
    info_dict['flop_info'] = flop_info 
    info_dict['activation_info'] = activation_info 
    print(len(param_info), info_dict['layer_num']) 
    np.save('frozen/{}/{}.npy'.format(arch_info[0], 'app_info'), info_dict)

# 42 41
# 34 33
# 108 107
# 116 115
# 130 129