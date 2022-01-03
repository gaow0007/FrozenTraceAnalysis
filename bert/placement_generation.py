import os, sys
import csv 
import numpy as np 
import torch.nn as nn 
sys.path.insert(0, './') 
from model import MLMTask
from adaptdl.torch.utils.misc import collect_atomic_layer_num 


def info_loader(): 
    for a1 in range(5): 
        for a2 in range(5): 
            for a3 in range(5): 
                for a4 in range(5): 
                    yield [a1, a2, a3, a4]

def isLegal(placement): 
    for i in range(3): 
        if placement[i] < placement[i+1]: 
            return False
    return True 

# prepare arch info 
arch_list = ['bert'] 
local_list = [8, 16, 24, 32, 48, 64] 
arch_info = dict() 
for arch in arch_list: 
    arch_info[arch] = list() 
    net = MLMTask(
        28784, 768, 12,
        3072, 12, 0.2).cuda()
    tot_layer_num = collect_atomic_layer_num(net)
    profile_tot_layer = int(tot_layer_num * 0.7) 
    arch_info[arch] = sorted(arch_info[arch])
    frozen_list = np.linspace(0, profile_tot_layer, 10)
    frozen_list = [int(layer) for layer in frozen_list] 
    for batch in local_list: 
        for frozen in frozen_list: 
            arch_info[arch].append((batch, frozen)) 


prefix = 'placement,local_bsz,step_time,sync_time,frozen_layer'
for app in [('WikiText2-bert', 'bert')]: 
    reference_info = None
    if len(app) == 2: 
        # fd = open('traces/{}/placements.csv'.format(app[1]), 'r')
        # reference_info = csv.DictReader(fd, delimiter=',')
        reference_info = None 

    if reference_info is None: 
        if not os.path.exists('frozen/{}'.format(app[0])): 
            os.makedirs('frozen/{}'.format(app[0]))

        filename = 'frozen/{}/placements.csv'.format(app[0])
        with open(filename, 'w') as f:
            f.write(prefix+'\n')
            for placement in info_loader(): 
                if isLegal(placement): 
                    while 0 in placement: 
                        placement.remove(0) 
                    if len(placement) <= 0: 
                        continue 
                    placement_str = '-'.join([str(pm) for pm in placement])
                    arch = app[0].split('WikiText2-')[1] 
                    for (batch, frozen) in arch_info[arch]: 
                        filename = '/mnt/lustre/wgao/workspace/adaptdl/examples/BERT/speed/model_WikiText2_placement_{}_bs_{}_placement_{}_frozen_{}.npy'.format(placement_str, batch, placement_str, frozen)
                        if not os.path.exists(filename): 
                            print(filename)
                            continue 
                        local_bsz = batch 
                        frozen_layer = frozen 
                        stats = np.load(filename, allow_pickle=True).tolist() 
                        step_time = None
                        sync_time = None
                        
                        for key, value in stats.items(): 
                            if 'metric' in key: 
                                profile = value[0].profile
                                for i_val in profile.values(): 
                                    optim_count = i_val['optim_count']
                                    step_time = i_val['optim_step_time'] / optim_count
                                    sync_time = i_val['optim_sync_time'] / optim_count
                        if step_time is None: 
                            # import pdb; pdb.set_trace() 
                            print(filename)

                        f.write('{},{},{},{},{}'.format(''.join([str(pm) for pm in placement]), local_bsz, step_time, sync_time, frozen_layer))
                        f.write("\n")


