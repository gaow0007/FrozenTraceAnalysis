import os, sys
import csv 
import numpy as np 

prefix = 'num_nodes,num_replicas,local_bsz,step_time,sync_time,frozen_layer'
for app in [('cifar10-ResNet18', 'cifar10')]: 
    reference_info = None
    if len(app) == 2: 
        fd = open('traces/{}/scalability.csv'.format(app[1]), 'r')
        reference_info = csv.DictReader(fd, delimiter=',')
    
    if reference_info is not None: 
        node_list = list() 
        replica_list = list()
        local_list = list() 
        step_list = list() 
        sync_list = list() 
        frozen_list = list() 
        param_info = [1.0, 0.9998453547631538, 0.9998338995604245, 0.9965348011743731, 0.9965233459716437, 0.9932242475855922, 0.993212792382863, 0.9899136939968115, \
            0.9899022387940821, 0.9866031404080308, 0.9865916852053014, 0.9799934884331986, 0.9799705780277399, 0.9667741844835341, 0.9667512740780755, \
            0.9660181411033973, 0.9659952306979387, 0.9527988371537329, 0.9527759267482743, 0.9395795332040685, 0.9395566227986099, 0.9131638357101984, \
            0.913118014899281, 0.8603324407224582, 0.8602866199115408, 0.8573540880128284, 0.857308267201911, 0.8045226930250882, 0.8044768722141707, \
            0.7516912980373479, 0.7516454772264305, 0.6460743288727848, 0.64598268725095, 0.43484039054365853, 0.4347487489218238, \
            0.4230186213269742, 0.4229269797051395, 0.21178468299784803, 0.2116930413760133, 0.0005507446687218476, 0.00045910304688701054]
        flop_info = [1, 0.9968212213483002, 0.9965857562629892, 0.9287718116933946, 0.9285363466080835, 0.8607224020384889, 0.8604869369531779, \
            0.7926729923835834, 0.7924375272982722, 0.7246235827286777, 0.7243881176433666, 0.6904811453585693, 0.6903634128159137, \
            0.6225494682463192, 0.6224317357036636, 0.6186642943386862, 0.6185465617960306, 0.550732617226436, 0.5506148846837804, \
            0.48280094011418595, 0.4826832075715304, 0.44877623528673305, 0.44871736901540527, 0.3809034244458107, 0.38084455817448293, \
            0.3770771168095055, 0.3770182505381777, 0.30920430596858317, 0.3091454396972554, 0.24133149512766083, 0.24127262885633305, 0.20736565657153583, \
            0.207336223435872, 0.13952227886627744, 0.1394928457306136, 0.13572540436563618, 0.13569597122997235, 0.06788202666037779, 0.06785259352471396, \
            3.864895511940247e-05, 9.215819455565821e-06]

        if not os.path.exists('frozen/{}'.format(app[0])): 
            os.makedirs('frozen/{}'.format(app[0]))
        filename = 'frozen/{}/scalability.csv'.format(app[0])
        with open(filename, 'w') as f:
            f.write(prefix+'\n')
            for info in reference_info: 
                for frozen_idx, frozen_ratio in enumerate(param_info): 
                    if frozen_idx % 4 == 0: 
                        node_list.append(info['num_nodes'])
                        replica_list.append(info['num_replicas'])
                        local_list.append(info['local_bsz'])
                        step_list.append(float(info['step_time']) * flop_info[frozen_idx])
                        sync_list.append(float(info['sync_time']) * param_info[frozen_idx]) 
                        frozen_list.append(frozen_idx)
            for num_node, num_replica, local_bsz, step_time, sync_time, frozen_layer in zip(node_list, replica_list, local_list, step_list, sync_list, frozen_list): 
                f.write('{},{},{},{},{},{}'.format(num_node, num_replica, local_bsz, step_time, sync_time, frozen_layer))
                f.write("\n")
            




