# %Header File Start-----------------------------------------------------------
#  Confidential（Unclassified）
#  COPYRIGHT (C) Sun Yat-sen University
#  THIS FILE MAY NOT BE MODIFIED OR REDISTRIBUTED WITHOUT THE
#  EXPRESSED WRITTEN CONSENT OF SYSU
# 
# %-----------------------------------------------------------------------------
#  Title   : train.py
#  Author  : Zhang wentao; 
#  E-mail  : z1282429194@163.com
#  Created : 07/15/2021
#  Description: moedl train 
# 
# %-----------------------------------------------------------------------------
#  Modification History:
#  V1.0: 2021.07.15, first created by Zhang wentao
#  V2.0: 2021.07.27, first modified by Zhang wentao
#  V2.1: 2021.08.08, second modified by Zhang wentao
#
# %Header File End--------------------------------------------------------------
import torch
import time
from Utils.cal_eval import AverageMeter, ProgressMeter, accuracy



def train_refine(Epoch, args, dataloader):
    
    args.net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):

        inputs, targets = inputs.to(args.device), targets.to(args.device)
        outputs = args.net(inputs)
        loss = args.criterion(outputs, targets)

        args.optimizer.zero_grad()
        loss.backward()
        args.optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        training_loss = (train_loss/(batch_idx+1))
        training_acc  = (correct/total)
        if batch_idx % args.print_freq == 0:
            args.logger.info('===== Train =====> Epoch: [{}/{}]    training_loss = {:.5f}    training_acc = {:.5f}    training_batchsize = {}'.format(Epoch, args.end_epoch-1, training_loss, training_acc, (args.trainbs)))
        
        
          
    # Record loss from the train run into the writer
    args.writer.add_scalar('Train/Loss', train_loss, Epoch)
    args.writer.flush()