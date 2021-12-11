# %Header File Start-----------------------------------------------------------
#  Confidential（Unclassified）
#  COPYRIGHT (C) Sun Yat-sen University
#  THIS FILE MAY NOT BE MODIFIED OR REDISTRIBUTED WITHOUT THE
#  EXPRESSED WRITTEN CONSENT OF SYSU
# 
# %-----------------------------------------------------------------------------
#  Title   : load_model.py
#  Author  : Zhang wentao; 
#  E-mail  : z1282429194@163.com
#  Created : 07/18/2021
#  Description: This is to load trained model from checkpoint file. 
# 
# %-----------------------------------------------------------------------------
#  Modification History:
#  V1.0: 2021.07.18, first created by Zhang wentao
# 
# %Header File End--------------------------------------------------------------
import torch
import os
import argparse

def load_model(args):
    # assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.save_path)
    args.net.load_state_dict(checkpoint['net'])
    args.best_acc = checkpoint['acc']
    args.start_epoch = checkpoint['epoch'] 
    args.optimizer.load_state_dict(checkpoint['optimizer'])