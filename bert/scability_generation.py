import os, sys
import csv 
import numpy as np 
import torch.nn as nn 
sys.path.insert(0, './') 
from model import MLMTask
from adaptdl.torch.utils.misc import collect_atomic_layer_num 


def info_loader():
    node_list = [8, 12, 16]
    for num_nodes in node_list:
        replica_list = [6, 8, 12, 16]
        for num_replicas in [num_nodes * i for i in range(1, 5)]: 
            if num_replicas not in replica_list:
                replica_list.append(num_replicas)
        for num_replicas in replica_list:
            if num_nodes > num_replicas:
                continue
            yield num_nodes, num_replicas



filename_list = list()
for filename in os.listdir('speed'):
    filename_list.append(filename)

def notFound(ident, filename_list):
    for filename in filename_list:
        if ident in filename:
            return False
    return True


def notFullFound(arch, placement, arch_info, filename_list):
    for (batch, frozen) in arch_info:
        ident = 'model_{}_placement_{}_bs_{}_placement_{}_frozen_{}.npy'.format(arch, placement, batch, placement, frozen)
        if ident not in filename_list:
            return True
    return False

def checkIfFound(arch, ref_num_nodes, ref_num_replicas, filename_list):
    legal_placement_list = list()
    for filename in filename_list:
        placement_str = filename.split('placement_')[-1].split('_frozen')[0]
        num_nodes = len(placement_str.split("-"))
        num_replicas = sum([int(gpu) for gpu in placement_str.split("-")])
        if num_nodes == ref_num_nodes and num_replicas == ref_num_replicas and placement_str not in legal_placement_list:
            legal_placement_list.append(placement_str)

    for placement in legal_placement_list:
        if not notFullFound(arch, placement, arch_info[arch], filename_list):
            return placement
    return None 

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


prefix = 'num_nodes,num_replicas,local_bsz,step_time,sync_time,frozen_layer'
for app in [('WikiText2-bert', 'bert')]: 
    reference_info = None
    if len(app) == 2: 
        # fd = open('traces/{}/placements.csv'.format(app[1]), 'r')
        # reference_info = csv.DictReader(fd, delimiter=',')
        reference_info = None 

    if reference_info is None: 
        if not os.path.exists('frozen/{}'.format(app[0])): 
            os.makedirs('frozen/{}'.format(app[0]))

        filename = 'frozen/{}/scalability.csv'.format(app[0])
        with open(filename, 'w') as f:
            f.write(prefix+'\n')
            for placement_info in info_loader(): 
                num_node, num_replica = placement_info
                placement_str = checkIfFound(arch, ref_num_nodes=num_node, ref_num_replicas=num_replica, filename_list=filename_list)
                if True: 
                    arch = app[0].split('WikiText2-')[1] 
                    for (batch, frozen) in arch_info[arch]: 
                        filename = '../BERT/speed/model_WikiText2_placement_{}_bs_{}_placement_{}_frozen_{}.npy'.format(placement_str, batch, placement_str, frozen)
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

                        f.write('{},{},{},{},{},{}'.format(num_node, num_replica, local_bsz, step_time, sync_time, frozen_layer))
                        f.write("\n")


