# %Header File Start-----------------------------------------------------------
#  Confidential（Unclassified）
#  COPYRIGHT (C) Sun Yat-sen University
#  THIS FILE MAY NOT BE MODIFIED OR REDISTRIBUTED WITHOUT THE
#  EXPRESSED WRITTEN CONSENT OF SYSU
# 
# %-----------------------------------------------------------------------------
#  Title   : weight_init.py
#  Author  : Zhang wentao; 
#  E-mail  : z1282429194@163.com
#  Created : 07/30/2021
#  Description: different ResNet models 
# 
# %-----------------------------------------------------------------------------
#  Modification History:
#  V1.0: 2021.08.04, first created by Zhang wentao
#  
#
# %Header File End--------------------------------------------------------------

import torch
import torch.nn as nn

def weight_init(Net):
    class_name = Net.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.kaiming_normal_(Net.weight.data)
    elif class_name.find('BatchNorm') != -1 and len(Net.weight.shape) >1:
        nn.init.kaiming_normal_(Net.weight.data)
        nn.init.constant(Net.weight.bias)