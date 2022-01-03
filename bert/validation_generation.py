import os, sys
import csv 
import numpy as np 
from adaptdl.torch.utils.misc import collect_atomic_layer_num 
from adaptdl.torch.layer_info import get_all_layer_info
sys.path.insert(0, './') 
import torch.nn as nn
from model import MLMTask

def patternMatch(epoch_list, cur_progress_list, ref_progress_list): 
    if len(cur_progress_list) == len(ref_progress_list): 
        return epoch_list 
        
    # import pdb; pdb.set_trace() 
    assert len(epoch_list) == len(cur_progress_list)
    selected_epoch_list = list() 
    for ref_progress in ref_progress_list: 
        selected_epoch = -1 
        close = sys.maxsize
        for epoch, cur_progress in zip(epoch_list, cur_progress_list): 
            if abs(ref_progress - cur_progress) < close and epoch not in selected_epoch_list: 
                close = abs(ref_progress - cur_progress)
                selected_epoch = epoch 
        assert selected_epoch != -1
        selected_epoch_list.append(selected_epoch)
    return selected_epoch_list 


def parse_efficiency_log(filename): 
    with open(filename, 'r') as f: 
        lines = f.readlines() 

    stats_info = list() 
    for idx, line in enumerate(lines):  
        iter_stats = eval(line)
        iter_stats['iteration'] = idx 
        stats_info.append(iter_stats)
    return stats_info 


def search(ref_progress, iteration_stats_info): 
    gap = sys.maxsize
    selected_iteration, selected_progress = -1, -1 
    for iteration, stats_info in enumerate(iteration_stats_info): 
        if abs(stats_info['progress'] - ref_progress) < gap: 
            selected_iteration = iteration 
            selected_progress = stats_info['progress']
            gap = abs(stats_info['progress'] - ref_progress)
    return selected_iteration + 1, selected_progress
    


for app in [('WikiText2-bert', 'bert')]: 
    reference_info = None
    ref_progress_list = None
    target_batch_size = 32 #  256, 512, 1024, 2048,
    model = MLMTask(
        28784, 768, 12,
        3072, 12, 0.2) 
    
    key_list = ['progress' ,'iteration', 'metric', 'grad_sqr', 'grad_var']
    for batch_size in [32, 64, 128, 256, 512, 1024]: 
        if len(app) == 2: 
            reference_info = None 
        dataset = app[0].split('-')[0] 
        arch =  app[0].split('-')[1] 
        num_replicas = batch_size // target_batch_size 
        epoch_stats_info = np.load('stats/{}/model_{}_bs_{}'.format(dataset, app[0].split('-')[0], batch_size * 128), allow_pickle=True).tolist()
        log_file = os.path.join('stats/{}/{}/efficiency.txt'.format(dataset, num_replicas))
        iteration_stats_info = parse_efficiency_log(log_file) 
        if batch_size == target_batch_size: 
            ref_progress_list = epoch_stats_info['progress'] 
            
        if reference_info is None: 
            node_list = list() 
            replica_list = list()
            local_list = list() 
            step_list = list() 
            sync_list = list() 
            
            if not os.path.exists('frozen/{}'.format(app[0])): 
                os.makedirs('frozen/{}'.format(app[0]))
            filename = 'frozen/{}/validation-{}.csv'.format(app[0], batch_size)
            if True: 
                for key in epoch_stats_info.keys(): 
                    if key.startswith('layer_'): 
                        key_list.append(key) 
                # import pdb; pdb.set_trace() 

            with open(filename, 'w') as f:
                for idx, key in enumerate(key_list): 
                    if idx == 0: 
                        f.write('{}'.format(key))
                    else:
                        f.write(',{}'.format(key))
                f.write('\n')
                # mport pdb; pdb.set_trace() 
                # for epoch in patternMatch(stats_info['epoch'], stats_info['progress'], ref_progress_list):  

                for epoch, ref_progress in enumerate(ref_progress_list): 
                    iteration, progress = search(ref_progress, iteration_stats_info)
                    epoch = int(iteration / len(iteration_stats_info) * len(epoch_stats_info['epoch'])) - 1
                    ratio = iteration / len(iteration_stats_info) * len(epoch_stats_info['epoch']) - epoch - 1
                    # import pdb; pdb.set_trace() 
                    grad_var_sum = 0
                    grad_sqr_sum = 0 
                    for key in epoch_stats_info.keys():
                        if key.startswith('layer_') and 'grad_var' in key: 
                            grad_var_sum += epoch_stats_info[key][epoch] * (1-ratio)
                            if ratio > 0 : 
                                grad_var_sum += epoch_stats_info[key][epoch + 1] * ratio
                        if key.startswith('layer_') and 'grad_sqr' in key: 
                            grad_sqr_sum += epoch_stats_info[key][epoch] * (1-ratio)
                            if ratio > 0 : 
                                grad_sqr_sum += epoch_stats_info[key][epoch + 1] * ratio 
                    
                    for idx, key in enumerate(key_list):  
                        if key == 'grad_var': 
                            val = abs(grad_var_sum)
                        elif key == 'grad_sqr': 
                            val = abs(grad_sqr_sum)
                        elif key == 'progress': 
                            val = progress 
                        elif key == 'iteration': 
                            val = iteration 
                        else: 
                            val = epoch_stats_info[key][epoch] * (1-ratio) 
                            if ratio > 0: 
                                val += epoch_stats_info[key][epoch+1] * ratio
                            val = abs(val)
                        # if 'grad_var' in key: 
                        #     val = abs(float(reference['grad_var']) * float(stats_info[key][epoch]) / grad_var_sum)
                        # elif 'grad_sqr' in key: 
                        #     val = abs(float(reference['grad_sqr']) * float(stats_info[key][epoch]) / grad_sqr_sum)
                        
                        if idx == 0: 
                            f.write('{}'.format(val))
                        else:
                            f.write(',{}'.format(val))
                    f.write('\n')
