import adaptdl 
import adaptdl.torch
from adaptdl.torch.utils.misc import collect_atomic_layer_num 
from adaptdl.torch.layer_info import get_all_layer_info

import numpy as np 
from models import * 
import matplotlib.pyplot as plt 


import autograd
import numpy as np
import collections
import scipy.optimize
import scipy.stats
import itertools 


MemParams = collections.namedtuple("MemParams", [
    # M_model ~ alpha_m + beta_m * local_bsz * frozen_ratio + 
    "alpha_m",  # Constant term of model memory 
    "beta_a",   # Multiplicative factor of activation memory 
    "beta_p",   # Multiplicative factor of parameter memory 
    "gamma",    # model the overlap between parameter and activation memory optimization 
])

def _rmse(pred, true):
    return np.sqrt((((pred - true)) ** 2).mean())

def _predict_memory(params, batch_size, activation_size, param_size):
    params = MemParams(*params)
    gamma = params.gamma
    # return params.alpha_m + ((params.beta_a * activation_size * batch_size) ** gamma  + (params.beta_p * param_size)**gamma) ** (1 / gamma)
    return params.alpha_m + params.beta_a * activation_size * batch_size

def _mem_obj_fn(params, batch_size, activation_size, param_size, memory):
    params = MemParams(*params)
    pred_memory = _predict_memory(params, batch_size, activation_size, param_size)
    # err1 = _rmse(np.log(pred_memory), np.log(memory))
    err1 = _rmse(pred_memory, memory)
    reg1 = 1e-3 * (params.gamma - 1) ** 2
    return err1 + reg1


def fit_mem_params(batch_size, activation_size, param_size, memory): 
    global np  # Replace numpy from autograd.
    orig_np = np
    np = autograd.numpy

    batch_size = np.array(batch_size) 
    activation_size = np.array(activation_size)
    param_size = np.array(param_size) 

    memory = np.array(memory) 
    # Set initial params to reasonable values.
    params = [0, 1, 1e-3, 1]
    # Set lower/upper bounds for each parameter. Add a small slack to lower
    # bounds to avoid numerical instability issues.
    lower = [0, 0, 0, 1]
    upper = [np.inf, np.inf, np.inf, np.inf]
    bounds = scipy.optimize.Bounds(lower, upper, keep_feasible=True) 
    args = (batch_size, activation_size, param_size, memory) 
    
    grad_fn = autograd.grad(_mem_obj_fn)
    result = scipy.optimize.minimize(_mem_obj_fn, params, args=args,
                                     jac=grad_fn, bounds=bounds)
    params = result.x
    np = orig_np  # Restore original numpy.
    return MemParams(*params)

def predict_max_local_bz(params, activation_size, param_size, memory_limit, batch_range): 
    left, right = batch_range
    while left < right - 1: 
        batch_size = (left + right) // 2
        memory = _predict_memory(params, batch_size, activation_size, param_size)
        print('predict batch {}, memory {}'.format(batch_size, memory))
        if memory > memory_limit: 
            right = batch_size - 1
        else:
            left = batch_size
    for batch_size in range(right, left - 1, -1): 
        memory = _predict_memory(params, batch_size, activation_size, param_size)
        if memory <= memory_limit: 
            return batch_size
    return None 


for arch_info in [('cifar10-GoogLeNet', 'GoogLeNet')]: # [('cifar10-ResNet18', 'ResNet18'), ('cifar10-VGG19', 'VGG19'), ('cifar10-ResNet50', 'ResNet50'), ('cifar10-MobileNetV2', 'MobileNetV2'), ('cifar10-GoogLeNet', 'GoogLeNet')]:
    info_dict = dict() 
    model = VGG('VGG19') if arch_info[1] == 'VGG19' else eval(arch_info[1])() 
    info_dict['layer_num'] = collect_atomic_layer_num(model) 
    flop_info, param_info, activation_info = get_all_layer_info(model, torch.randn(1, 3, 32, 32))
    info_dict['param_info'] = param_info
    info_dict['flop_info'] = flop_info 
    # print(len(param_info), info_dict['layer_num']) 
    frozen_list = list() 
    batch_list = list() 
    if True: 
        memory_stats = np.load('../pytorch-cifar/memory/model_{}.npy'.format(arch_info[1]), allow_pickle=True).tolist() 
        for key, value in memory_stats.items():  
            if not isinstance(key, str):  
                frozen_list.append(key)
                for batch in value.keys(): 
                    if batch not in batch_list: 
                        batch_list.append(batch)
        
        
        frozen_list = sorted(frozen_list)
        batch_list = sorted(batch_list)  
        # print(frozen_list)
        # print(batch_list)
        # print(arch_info)

        # train_list = [(0, 257), (0, 182), (0, 363), (4, 257), (4, 182), (4, 363), (8, 257), (8, 182), (8, 363)]
        # train_list = [(0, 725), (0, 513), (0, 363), (4, 257), (4, 182), (4, 363), (8, 257), (8, 182), (8, 363)]
        train_list = [(frozen, batch) for frozen, batch in itertools.product(frozen_list[:1], batch_list)]
        print(train_list)
        # import pdb; pdb.set_trace() 
        train_batch_size = list() 
        train_activation_size = list() 
        train_param_size = list() 
        train_memory = list() 
        for (frozen, batch) in train_list: 
            train_batch_size.append(batch) 
            train_activation_size.append(activation_info[frozen] * activation_info[-1])
            train_param_size.append(param_info[frozen] * param_info[-1])
            train_memory.append(memory_stats[frozen][batch])

        params = fit_mem_params(train_batch_size, train_activation_size, train_param_size, train_memory)

        # for frozen_layer in frozen_list: 
        for frozen_layer in [0]: 
            print('actual memory {}, total memory {}'.format(memory_stats[0][513], memory_stats['total_memory']))
            batch_limit = predict_max_local_bz(params, activation_info[frozen], param_info[frozen], memory_stats['total_memory'], (513, 2048))
            print('frozen {}, arch {}, batch limit {}'.format(frozen_layer, arch_info[1], batch_limit))
        continue 
        for batch in batch_list[-1:]: 
            memory_list = list()  
            estimate_list = list() 
            for frozen_layer in frozen_list: 
                memory_list.append(memory_stats[frozen_layer][batch]) 
                estimate_memory = _predict_memory(params, batch, activation_info[frozen_layer], param_info[frozen_layer]) 
                estimate_list.append(estimate_memory)
            print(memory_list)
            print(estimate_list)
            plt.plot(frozen_list, memory_list, marker='*', label='gt')
            plt.plot(frozen_list, estimate_list, color='r', label='estimate')
            print('plot {}, arch {}'.format(batch, arch_info[1]))
        plt.legend() 
        plt.savefig('show/{}.jpg'.format(arch_info[1]))
        plt.cla() 
        continue 
        exit(0)
        for batch in batch_list[-1:]:
            memory_list = list()  
            estimate_list = list() 
            for frozen_layer in frozen_list: 
                memory_list.append(memory_stats[frozen_layer][batch])
                # import pdb; pdb.set_trace() 
                ratio = (activation_info[frozen_layer] * activation_info[-1] * batch + param_info[frozen_layer] * param_info[-1])  / \
                    (activation_info[0] * activation_info[-1] * batch + param_info[0] * param_info[-1])
                print(ratio, activation_info[frozen_layer])
                estimate_list.append(memory_stats[0][batch] * activation_info[frozen_layer])

            plt.plot(frozen_list, memory_list, marker='*')
            plt.plot(frozen_list, estimate_list, color='r')
            print('plot {}, arch {}'.format(batch, arch_info[1]))
        plt.savefig('show/{}.jpg'.format(arch_info[1]))
        plt.cla() 
