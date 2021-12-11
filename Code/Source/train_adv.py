# %Header File Start-----------------------------------------------------------
#  Confidential（Unclassified）
#  COPYRIGHT (C) Sun Yat-sen University
#  THIS FILE MAY NOT BE MODIFIED OR REDISTRIBUTED WITHOUT THE
#  EXPRESSED WRITTEN CONSENT OF SYSU
# 
# %-----------------------------------------------------------------------------
#  Title   : train_adv.py
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



def train_adv(Epoch, args, dataloader):
    args.logger.info('Epoch: {}'.format(Epoch))
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.3f')
    top5 = AverageMeter('Acc@5', ':6.3f')
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(Epoch))


    args.net.train()
    end = time.time()
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs, targets = inputs.to(args.device), targets.to(args.device)

        # obtain adv samplers
        atk_data = args.atk.perturb(inputs, targets)

        # predicted results of clean samplers
        clean_outputs = args.net(inputs)

        # predicted results of adv samplers
        adv_outputs = args.net(atk_data)
        
        # cal two parts loss: loss_clean, loss_adv
        loss_clean = args.criterion(clean_outputs, targets)
        loss_adv = args.criterion(adv_outputs, targets)

        # obtain adv training loss
        loss = loss_clean + loss_adv

        args.optimizer.zero_grad()
        loss.backward()
        args.optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        
        # measure accuracy and record loss
        acc1, acc5 = accuracy(adv_outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))
        if batch_idx % args.print_freq == 0:
            args.logger.info(progress.display(batch_idx))    
    # Record loss from the train run into the writer
    args.writer.add_scalar('Train/Loss', losses.avg, Epoch)
    args.writer.flush()