import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import adaptdl
import adaptdl.torch
# from apex import amp
# from apex.amp._amp_state import _amp_state
from torch.utils.tensorboard import SummaryWriter
from adaptdl.torch.utils.misc import collect_atomic_layer_num, apply_frozen, cal_frozen_layer, frozen_net
import numpy as np 
import copy 


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', type=str, default='/mnt/lustre/share/zhangwenwei/data/imagenet/',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18', 
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--autoscale-bsz', dest='autoscale_bsz', default=False, action='store_true', help='autoscale batchsize')
parser.add_argument('--placement', required=True, type=str, help='placement')
parser.add_argument('--max_bs', default=1024, type=int, help='number of epochs')

def set_local_batch_size(local_batch_size): 
    batch_size = local_batch_size * adaptdl.env.num_replicas() 
    os.environ['TARGET_BATCH_SIZE'] = "{}".format(batch_size) 

def getCurTime():
    torch.cuda.synchronize(0)
    return time.time() 

def walkover(trainloader): 
    for idx, (_, _) in enumerate(trainloader):  
        print(idx, len(trainloader))
        pass



def main_worker(args):

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch == 'GoogLeNet': 
        model = models.__dict__['googlenet']().cuda()
    elif args.arch == 'MobileNetV2':
        model = models.__dict__['mobilenet_v2']().cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    adaptdl.torch.init_process_group("nccl")
    # model, optimizer = amp.initialize(model, optimizer) 
    adaptdl.torch.layer_info.profile_layer_info(model, torch.randn(1, 3, 224, 224)) 
    model = adaptdl.torch.AdaptiveDataParallel(model, optimizer, find_unused_parameters=True) 

    cudnn.benchmark = True

    # Data loading code
    if adaptdl.env.num_replicas() > 16:
        traindir = os.path.join(args.data, 'train')
    else:
        traindir = os.path.join(args.data, 'val')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = adaptdl.torch.AdaptiveDataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = adaptdl.torch.AdaptiveDataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    tensorboard_dir = 'log_dir'
    tot_layer_num = collect_atomic_layer_num(model)
    profile_tot_layer = int(tot_layer_num * 0.9) # // 2
    frozen_list = np.linspace(0, profile_tot_layer, 10)
    frozen_list = [int(layer) for layer in frozen_list] 
    from adaptdl.torch._metrics import _metrics_state
    import collections 
    with SummaryWriter(tensorboard_dir) as writer:
        for epoch, fronzen_layer_num in enumerate(frozen_list):
            adaptdl.torch.set_current_frozen_layer(fronzen_layer_num)
            apply_frozen(model, fronzen_layer_num) 

            for local_batch_size in [20, 28, 40, 57, 81, 115, 163, 200]: 
                if local_batch_size > args.max_bs: 
                    continue 
                if adaptdl.env.replica_rank() == 0:
                    print('local_batch_size == {}, fronzen_layer_num == {}'.format(local_batch_size, fronzen_layer_num))
                
                epoch_frequency = 1
                training_time = 0
                time_list = list()
                metric_list = list() 
                metric_info = _metrics_state()
                info_dict = dict() 
                metric_info.profile = collections.defaultdict(collections.Counter) 
                path = 'speed/model_{}_placement_{}_bs_{}_placement_{}_frozen_{}.npy'.format(args.arch, args.placement, local_batch_size, args.placement, fronzen_layer_num)
                if os.path.exists(path): 
                    continue 

                train_loader = adaptdl.torch.AdaptiveDataLoader(
                    train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                    num_workers=args.workers, pin_memory=True)
                train_loader._elastic.train() 
                set_local_batch_size(local_batch_size=local_batch_size)
                for i in range(epoch_frequency):
                    start_time = getCurTime()
                    train(train_loader, model, criterion, optimizer, epoch, args, writer, limit=20)
                    print('run here')
                    time_list.append((getCurTime() - start_time))
                    training_time += time_list[-1]
                    metric_list.append(copy.deepcopy(_metrics_state())) 

                
                info_dict['freeze_layer'] = fronzen_layer_num
                info_dict['list_{}'.format(fronzen_layer_num)] = time_list
                info_dict['metric_{}'.format(fronzen_layer_num)] = metric_list 

                if adaptdl.env.replica_rank() == 0:
                    print('saving speed info')
                    with open('speed/model_{}_placement_{}_bs_{}_placement_{}_frozen_{}.npy'.format(args.arch, args.placement, local_batch_size, args.placement, fronzen_layer_num), 'wb') as f:
                        np.save(f, info_dict) 

            

def train(train_loader, model, criterion, optimizer, epoch, args, writer, limit):
    stats = adaptdl.torch.Accumulator()

    # switch to train mode
    model.train()
    frozen_net(model)
    stats = adaptdl.torch.Accumulator() 
    for i, (images, target) in enumerate(train_loader): 
        if i >= limit: 
            break 
        # measure data loading time

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)
        loss.backward() 
        model.adascale.step()

    with stats.synchronized():
        pass 


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append((correct_k * 100.0 / batch_size, correct_k.item(), batch_size))
        return res


if __name__ == '__main__':
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    main_worker(args)
