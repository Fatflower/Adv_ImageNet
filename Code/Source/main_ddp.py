# %Header File Start-----------------------------------------------------------
#  Confidential（Unclassified）
#  COPYRIGHT (C) Sun Yat-sen University
#  THIS FILE MAY NOT BE MODIFIED OR REDISTRIBUTED WITHOUT THE
#  EXPRESSED WRITTEN CONSENT OF SYSU
# 
# %-----------------------------------------------------------------------------
#  Title   : main.py
#  Author  : Zhang wentao; 
#  E-mail  : z1282429194@163.com
#  Created : 07/15/2021
#  Description: main program 
# 
# %-----------------------------------------------------------------------------
#  Modification History:
#  V1.0: 2021.07.15, first created by Zhang wentao
#  V2.0: 2021.07.24, first modified by Zhang wentao
#  V2.1: 2021.07.27, second modified by Zhang wentao
#  V2.1: 2021.08.12, third modified by Zhang wentao
# 
# %Header File End--------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder, CIFAR10
from torch.utils.data import DataLoader

# import DDP
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from torch.utils.tensorboard import SummaryWriter  

import argparse
import os

# set path
import sys
path = os.path.dirname(os.path.dirname(__file__)) 
os.chdir(path)
sys.path.append(path)

# import your own library
from Utils.log import log_creater
from Source.test_ddp import test_ddp
from Source.train_ddp import train_ddp
from Utils.load_model import load_model
from Utils.weight_init import weight_init
from Model.Resnet_refine import ResNet_refine
from Model.VGG_refine import VGG_refine
from Utils.transform_dataset import common_test_transfoms, common_train_transfoms
from Utils.tool_kits import makedirs, record_para
from Utils.cal_eval import AverageMeter, adjust_learning_rate, accuracy, ProgressMeter

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--a', type=str, default='resnet18', help='model architecture' )
parser.add_argument('--classes_num', default=1000, type=int, help='Number of classes')
parser.add_argument('--pretrained', action="store_false", help='Pre-training and no pre-training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--train', default=0, type=int, help='training or evaluation', choices = [0, 1])
parser.add_argument('--mean', default=[0.485, 0.456, 0.406], type=list, help='mean of dataset')
parser.add_argument('--std', default=[0.229, 0.224, 0.225], type=list, help='std of dataset')
parser.add_argument('--trainbs', default=256, type=int, help='trainloader batch size')
parser.add_argument('--testbs', default=256, type=int, help='testloader batch size')
parser.add_argument('--epoch', default=100, type=int, help='epoch of model training')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--seed', default=1, type=int, help='seed for initializing training. ')
parser.add_argument('--GPU', default='6,7', type=str, help='cuda_visible_devices')
parser.add_argument('--data_dir', default='./data', type=str, help='path of dataset ')
parser.add_argument('--checkpoint', default='../Checkpoint', type=str, help='path of checkpoint')
parser.add_argument('--info_log', default='train', type=str, help='title of log')
parser.add_argument('--dir_log', default='../Result/Log', type=str, help='path of log')
parser.add_argument('--dir_run', default='../Result/Runs', type=str, help='path of runs')

parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--epochs', default=9, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

args = parser.parse_args()

# set seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU





def main():
    # Log
    args.dir_log = os.path.join(args.dir_log, args.a)
    makedirs(args.dir_log)
    print(args.dir_log)
    info_log = args.info_log + ('{}'.format(args.classes_num))
    print(info_log)
    args.logger = log_creater(args.dir_log, info_log)

    # Record all hyperparameters and paths used 
    record_para(args)    

    # set path
    tmp_path = args.a + '_' +  info_log + '_ckpt.pth'
    args.logger.info(tmp_path)
    args.save_path = os.path.join(args.checkpoint, args.a)
    makedirs(args.save_path)
    args.save_path = os.path.join(args.save_path, tmp_path)



    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    
    
def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global randatasetsk among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.a[0:1] == 'r':
        model = ResNet_refine(args.a, args.pretrained, args.classes_num)
    else:
        model = VGG_refine(args.a, args.pretrained, args.classes_num)
    
    if args.pretrained == False:
        model.apply(weight_init)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)


    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # optionally resume from a checkpoint
    if args.resume:
        if args.gpu is None:
            checkpoint = torch.load(args.save_path)
        else:
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.resume, map_location=loc)
        args.start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        if args.gpu is not None:
            # best_acc1 may be from a checkpoint from a different GPU
            best_acc1 = best_acc1.to(args.gpu)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
        

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
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

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        test_ddp(val_loader, model, criterion, args)
        return
    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train_ddp(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = test_ddp(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict()
                }
            torch.save(state, args.save_path)
            args.logger.info('epoch :{}    best_acc1 :{}'.format(epoch + 1, best_acc1))
            




if __name__ == '__main__':
    main()





