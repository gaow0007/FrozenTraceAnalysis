import adaptdl 
import adaptdl.torch
from adaptdl.torch.utils.misc import collect_atomic_layer_num 
from adaptdl.torch.layer_info import get_all_layer_info

import numpy as np 
from models import * 
import matplotlib.pyplot as plt 


prefix = 'frozen_layer,local_bsz,gpu_memory'

for arch_info in [('cifar10-ResNet18', 'ResNet18'), ('cifar10-VGG19', 'VGG19'), ('cifar10-ResNet50', 'ResNet50'), ('cifar10-MobileNetV2', 'MobileNetV2'), ('cifar10-GoogLeNet', 'GoogLeNet')]: 
    info_dict = dict() 
    model = VGG('VGG19') if arch_info[1] == 'VGG19' else eval(arch_info[1])() 
    info_dict['layer_num'] = collect_atomic_layer_num(model) 
    flop_info, param_info, activation_info = get_all_layer_info(model, torch.randn(1, 3, 32, 32))
    info_dict['param_info'] = param_info
    info_dict['flop_info'] = flop_info 
    print(len(param_info), info_dict['layer_num']) 
    frozen_list = list() 
    batch_list = list() 
    try: 
        memory_stats = np.load('../pytorch-cifar/memory/model_{}.npy'.format(arch_info[1]), allow_pickle=True).tolist() 
        filename = 'frozen/{}/memory.csv'.format(arch_info[0])
        with open(filename, 'w') as f:
            f.write(prefix+'\n')
            for key, value in memory_stats.items():  
                if not isinstance(key, str):  
                    frozen = key
                    for batch in value.keys(): 
                         f.write('{},{},{}'.format(frozen, batch,value[batch]))
                         f.write('\n')


        

        # for key, value in memory_stats.items():  
        #     if not isinstance(key, str):  
        #         frozen_list.append(key)
        #         for batch in value.keys(): 
        #             if batch not in batch_list: 
        #                 batch_list.append(batch)
        # print(arch_info)
        # frozen_list = sorted(frozen_list)
        # batch_list = sorted(batch_list)
        # for batch in batch_list[-1:]:
        #     memory_list = list()  
        #     estimate_list = list() 
        #     for frozen_layer in frozen_list: 
        #         memory_list.append(memory_stats[frozen_layer][batch]) 
        #         ratio = (activation_info[frozen_layer] * activation_info[-1] * batch + param_info[frozen_layer] * param_info[-1])  / \
        #             (activation_info[0] * activation_info[-1] * batch + param_info[0] * param_info[-1])
        #         print(ratio, activation_info[frozen_layer])
        #         estimate_list.append(memory_stats[0][batch] * activation_info[frozen_layer])

        #     plt.plot(frozen_list, memory_list, marker='*')
        #     plt.plot(frozen_list, estimate_list, color='r')
        #     print('plot {}, arch {}'.format(batch, arch_info[1]))
        # plt.savefig('show/{}.jpg'.format(arch_info[1]))
        # plt.cla() 

        continue
        for frozen_layer in frozen_list: 
            print('processing {} layer ...'.format(frozen_layer))
            batch1 = 513 # batch_list[0]
            batch2 = 725 # batch_list[1]
            delta = (memory_stats[frozen_layer][batch2] - memory_stats[frozen_layer][batch1]) / (batch2 - batch1)
            max_batch_size = (memory_stats['total_memory'] - memory_stats[frozen_layer][batch1]) / delta + batch1 
            print('estimate max batch size is {}'.format(max_batch_size))
            # for batch in batch_list[2:]:
            #     actual_mem = memory_stats[frozen_layer][batch]
            #     estimate_mem = memory_stats[frozen_layer][batch1] + delta * (batch - batch1)
            #     print('estimate error is {}'.format((actual_mem - estimate_mem) / actual_mem))
            # print(batch_list)
        # exit(0)

    except Exception as e: 
        print(e) 
    
    # np.save('frozen/{}/{}.npy'.format(arch_info[0], 'app_info'), info_dict)
