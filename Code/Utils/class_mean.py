# %Header File Start-----------------------------------------------------------
#  Confidential（Unclassified）
#  COPYRIGHT (C) Sun Yat-sen University
#  THIS FILE MAY NOT BE MODIFIED OR REDISTRIBUTED WITHOUT THE
#  EXPRESSED WRITTEN CONSENT OF SYSU
# 
# %-----------------------------------------------------------------------------
#  Title   : class_mean.py
#  Author  : Zhang wentao; 
#  E-mail  : z1282429194@163.com
#  Created : 07/15/2021
#  Description: moedl train 
# 
# %-----------------------------------------------------------------------------
#  Modification History:
#  V1.0: 2021.08.08, first created by Zhang wentao
#
# %Header File End--------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision.models as models
import torchvision.transforms as transforms
from Utils.tool_kits import distance_diff, tensor2numpy

import os
import argparse

def cal_classes_mean(args):
    for cnt in range(args.num_up):
        args.classes_mean[cnt] /= args.classes_cnt[cnt]

def recover_classes_mean(args):
    for cnt in range(args.num_up):
        args.classes_mean[cnt] *= args.classes_cnt[cnt]



