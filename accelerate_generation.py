import os, sys
import csv 
import numpy as np 

sys.path.insert(0, './')
from models import * 
import numpy as np 

def collect_atomic_layer_num(net):
    atomic_layer_num = 0
    for module in net.modules():
        if isAtomicLayer(module):
            atomic_layer_num += 1
    return atomic_layer_num


def PipeTransformerMethod(layer_num, alpha, epoch):
    second_term = 0.0
    for e in range(2, epoch + 1):
        second_term += ((layer_num * alpha) / pow(1 - alpha, e))
    return pow(1 - alpha, epoch) * ((layer_num * alpha) / (1 - alpha) + second_term)


def cal_frozen_layer(epoch, tot_epoch, tot_layer_num):
    fronzen_layer_num = int(PipeTransformerMethod(tot_layer_num, 1.0 / tot_epoch, epoch)) // 2 * 2 
    return fronzen_layer_num


def isAtomicLayer(mod):
    return (isinstance(mod, nn.Conv2d))  or isinstance(mod, nn.Linear) or isinstance(mod, nn.BatchNorm2d) # and mod.in_channels != 3

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
arch_list = ['ResNet18', 'GoogLeNet', 'VGG19', 'MobileNetV2', 'ResNet50'] 
# max_list = [1024, 513, 1024, 513, 513]
max_list = [1024, 513, 1024, 1024, 513]
local_list = [32, 45, 64, 91, 129, 182, 257, 363, 513, 725, 1024]
# local_list = [1024]
arch_info = dict() 
for arch in arch_list: 
    arch_info[arch] = list() 
    if arch != 'VGG19':
        net = eval(arch)()
    else:
        net = VGG('VGG19') 
    tot_layer_num = collect_atomic_layer_num(net)
    profile_tot_layer = int(tot_layer_num * 0.9) # // 2
    arch_info[arch] = sorted(arch_info[arch])
    frozen_list = np.linspace(0, profile_tot_layer, 10)
    frozen_list = [int(layer) for layer in frozen_list]# [-2:] # TODO 

    # frozen_list = [42, 48, 66]
    # frozen_list = [90]
    for batch in local_list: 
        if batch > max_list[arch_list.index(arch)]: continue 
        for frozen in frozen_list: 
            # print(arch, (batch, frozen))
            arch_info[arch].append((batch, frozen)) 



def fetch_time(stats): 
    optim_count = 0 
    step_time = 0 
    sync_time = 0
    for key, value in stats.items():
        if 'metric' in key: 
            profile = value[0].profile
            for i_val in profile.values():
                optim_count = i_val['optim_count']
                step_time = i_val['optim_step_time'] / optim_count
                sync_time = i_val['optim_sync_time'] / optim_count
                # print('optim_count == {}'.format(optim_count))
    assert sync_time > 0, 'should > 0'
    return step_time, sync_time, stats['abs_time']



prefix = 'placement,local_bsz,step_time,sync_time,frozen_layer'
for app in [('cifar10-ResNet18', 'cifar10'), ('cifar10-VGG19', 'cifar10'), ('cifar10-ResNet50', 'cifar10'), ('cifar10-MobileNetV2', 'cifar10'), ('cifar10-GoogLeNet', 'cifar10')]: 

    fd = open('frozen/{}/placements.csv'.format(app[0]), 'r')
    reference_info = csv.DictReader(fd, delimiter=',') 

    rewrite_stats = np.load('../pytorch-cifar/accelerate/model_{}_rewrite_True.npy'.format(app[0].split('-')[1]), allow_pickle=True).tolist() 
    base_stats = np.load('../pytorch-cifar/accelerate/model_{}_rewrite_False.npy'.format(app[0].split('-')[1]), allow_pickle=True).tolist() 
    # import pdb; pdb.set_trace() 
    # if reference_info is None: 
    if True: 
        if not os.path.exists('frozen/{}'.format(app[0])): 
            os.makedirs('frozen/{}'.format(app[0]))

        rewrite_accelerate = dict() 
        for (batch, frozen) in arch_info[app[0].split('-')[1]]: 
            if batch not in rewrite_accelerate: 
                rewrite_accelerate[batch] = dict() 
            rewrite_accelerate[batch][frozen] = None 
        # filename = 'frozen/{}/rewrite.csv'.format(app[0])
        # with open(filename, 'w') as f:
        #     f.write(prefix+'\n')
        if True: 
            placement = '1'
            placement_str = '-'.join([str(pm) for pm in placement])
            arch = app[0].split('cifar10-')[1]
            for (batch, frozen) in arch_info[arch]: 
                # filename = '../pytorch-cifar/speed/model_{}_placement_{}_bs_{}_placement_{}_frozen_{}.npy'.format(arch, placement_str, batch, placement_str, frozen)
                # if not os.path.exists(filename): 
                #     print(filename)
                #     continue 
                local_bsz = batch 
                frozen_layer = frozen 
                # stats = np.load(filename, allow_pickle=True).tolist() 
                stats = base_stats[frozen][batch]
                step_time = None
                sync_time = None
                # print(rewrite_stats[frozen].keys())
                rewrite_speed = rewrite_stats[frozen][batch]
                step_time, sync_time, abs_time = fetch_time(stats)
                rewrite_step_time, rewrite_sync_tme, rewrite_abs_time = fetch_time(rewrite_speed)
                # print('unfused', step_time, sync_time, abs_time)
                # print('fused', rewrite_step_time, rewrite_sync_tme, rewrite_abs_time)
                # import pdb; pdb.set_trace() 
                # print('app {}, batch {}, frozen {}, rewrite accelerate {}'.format(app, batch, frozen, (abs_time ) / (rewrite_abs_time )))
                rewrite_accelerate[batch][frozen] = step_time / rewrite_step_time 
    # print(rewrite_accelerate.keys())
    if reference_info is not None: 
        filename = 'frozen/{}/placements-rewrite.csv'.format(app[0])
        with open(filename, 'w') as f:
            f.write(prefix+'\n')
            # prefix = 'placement,local_bsz,step_time,sync_time,frozen_layer'
            for reference in reference_info: 
                placement = reference['placement']
                local_bsz = reference['local_bsz']
                batch = local_bsz 
                frozen = reference['frozen_layer']
                step_time = float(reference['step_time']) / rewrite_accelerate[int(batch)][int(frozen)]
                sync_time = float(reference['sync_time']) * (1 + np.random.randn() * 1e-3)
                f.write('{},{},{},{},{}'.format(placement, local_bsz, step_time, sync_time, frozen_layer))
                f.write("\n")

