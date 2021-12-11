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
from torchvision.datasets import ImageFolder, CIFAR10
from torch.utils.data import DataLoader

# set
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
from Source.test import test
from Source.train import train
from Source.test_refine import test_refine
from Source.train_refine import train_refine
from Source.test_adv import test_adv
from Source.train_adv import train_adv
from Utils.load_model import load_model
from Utils.weight_init import weight_init
from Model.Resnet_refine import ResNet_refine
from Model.VGG_refine import VGG_refine
from Utils.transform_dataset import common_test_transfoms, common_train_transfoms
from Utils.tool_kits import makedirs, record_para

# set attack
from advertorch.attacks import LinfPGDAttack

# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--a', type=str, default='resnet18', help='model architecture' )
parser.add_argument('--classes_num', default=10, type=int, help='Number of classes')
parser.add_argument('--pretrained', action="store_false", help='Pre-training and no pre-training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--train', default=0, type=int, help='training or evaluation', choices = [0, 1])
parser.add_argument('--mean', default=[0.485, 0.456, 0.406], type=list, help='mean of dataset')
parser.add_argument('--std', default=[0.229, 0.224, 0.225], type=list, help='std of dataset')
parser.add_argument('--trainbs', default=256, type=int, help='trainloader batch size')
parser.add_argument('--testbs', default=256, type=int, help='testloader batch size')
parser.add_argument('--epoch', default=150, type=int, help='epoch of model training')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--seed', default=1, type=int, help='seed for initializing training. ')
parser.add_argument('--gpu', default='6,7', type=str, help='cuda_visible_devices')
parser.add_argument('--data_dir', default='./data', type=str, help='path of dataset ')
parser.add_argument('--checkpoint', default='../Checkpoint', type=str, help='path of checkpoint')
parser.add_argument('--info_log', default='train', type=str, help='title of log')
parser.add_argument('--dir_log', default='../Result/Log', type=str, help='path of log')
parser.add_argument('--dir_run', default='../Result/Runs', type=str, help='path of runs')

args = parser.parse_args()

# set seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

args.best_acc = 0
args.best_acc_epoch = 0

# Log
args.dir_log = os.path.join(args.dir_log, args.a)
makedirs(args.dir_log)
print(args.dir_log)
info_log = args.info_log + ('{}'.format(args.classes_num))
print(info_log)
args.logger = log_creater(args.dir_log, info_log)

# Record all hyperparameters and paths used 
record_para(args)

# Load Data
image_size = (256, 256)
crop_size = (224, 224)


args.logger.info('==> Preparing data..')

traindir = os.path.join(args.data_dir, 'train')
valdir = os.path.join(args.data_dir, 'val')

trainset = ImageFolder(root=traindir, transform = common_train_transfoms(image_size, crop_size, args.mean, args.std))
args.logger.info('The number of trainset: {}'.format(len(trainset)))
args.trainloader = DataLoader(trainset, batch_size=args.trainbs, shuffle=True, num_workers=2)

testset = ImageFolder(root=valdir, transform = common_test_transfoms(image_size, crop_size, args.mean, args.std))
args.logger.info('The number of testset: {}'.format(len(testset)))
args.testloader = DataLoader(testset, batch_size=args.testbs, shuffle=False, num_workers=2)



# tensorboard
args.dir_run = os.path.join(args.dir_run, args.a)
makedirs(args.dir_run)
print(args.dir_run)
writer = SummaryWriter(args.dir_run)
args.writer = writer

# Load Model
args.logger.info('==> Loading the untrained model..')
if args.a[0:1] == 'r':
    net = ResNet_refine(args.a, args.pretrained, args.classes_num)
else:
    net = VGG_refine(args.a, args.pretrained, args.classes_num)

tmp_path = args.a + '_' +  info_log + '_ckpt.pth'
args.logger.info(tmp_path)
args.save_path = os.path.join(args.checkpoint, args.a)
makedirs(args.save_path)
print(args.save_path)
args.save_path = os.path.join(args.save_path, tmp_path)
print(args.save_path)

if args.pretrained == False:
    net.apply(weight_init)


args.net = net.to(args.device)


if torch.cuda.device_count() > 1:
    args.logger.info('Data is distributed on multiple CPUs in parallel: ' + args.gpu)
    args.net = torch.nn.DataParallel(args.net)
    cudnn.benchmark = True
elif torch.cuda.device_count() == 1: 
    args.logger.info('Data is distributed on a signal CPUs : ' + args.gpu)

args.criterion = nn.CrossEntropyLoss()

# args.optimizer = optim.SGD([
#     {'params': args.net.feature.parameters(), 'lr': 1e-3},
#     {'params': args.net.fc.parameters()}],
#     lr=args.lr, momentum=0.9, weight_decay=5e-4
#     )

args.optimizer = optim.SGD(args.net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(args.optimizer, step_size=70, gamma=0.1)

# set attack
if args.train == 0:
    Targeted = False
elif args.train == 1:
    Targeted = True

args.atk = LinfPGDAttack(args.net, loss_fn=nn.CrossEntropyLoss(reduction="sum"), 
    eps=8/255, nb_iter=10, eps_iter=2/255, rand_init=True, clip_min=-2.1179039301310043, clip_max=2.6399999999999997,
    targeted=Targeted)

args.start_epoch = 0 # start from epoch 0 or last checkpoint epoch

if args.resume:
    # Load best checkpoint trained last time.
    args.logger.info('==> Resuming from checkpoint..')
    load_model(args)


args.end_epoch = args.start_epoch + args.epoch

if args.train == 0:
    args.logger.info('Start training!')
    for Epoch in range(args.start_epoch, args.end_epoch):
        # adv train
        train_adv(Epoch, args, args.trainloader)
        test_adv(Epoch, args, args.testloader)

        # normal train
        # train(Epoch, args, args.trainloader)
        # test(Epoch, args, args.testloader)       
        #另一种train和test
        # train_refine(Epoch, args, args.trainloader)
        # test_refine(Epoch, args, args.testloader)

        scheduler.step()  
        # print(args.optimizer.param_groups[0]['lr'])

    args.logger.info("The accuracy at Epoch {} is best accuracy: {}".format(args.best_acc_epoch, args.best_acc))
    args.logger.info('Finish training!')
else:
    args.logger.info('==> Loading the trained model parameters from checkpoint..')
    load_model(args)
    args.logger.info('Start testing!')
    # test_adv(1, args)
    args.logger.info('Finish testing!')

args.writer.close()







