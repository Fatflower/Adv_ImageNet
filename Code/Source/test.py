# %Header File Start-----------------------------------------------------------
#  Confidential（Unclassified）
#  COPYRIGHT (C) Sun Yat-sen University
#  THIS FILE MAY NOT BE MODIFIED OR REDISTRIBUTED WITHOUT THE
#  EXPRESSED WRITTEN CONSENT OF SYSU
# 
# %-----------------------------------------------------------------------------
#  Title   : test.py
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
#  V2.2: 2021.08.12, third modified by Zhang wentao
#  V2.3: 2021.08.12, fourth modified by Zhang wentao
# %Header File End--------------------------------------------------------------
import torch
import os
from sklearn.metrics import confusion_matrix
import time
from Utils.cal_eval import AverageMeter, ProgressMeter, accuracy


def test(Epoch, args, dataloader):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.3f')
    top5 = AverageMeter('Acc@5', ':6.3f')
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')


    args.net.eval()
    
    # with torch.no_grad():
    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(dataloader):

        inputs, targets = inputs.to(args.device), targets.to(args.device)

        outputs = args.net(inputs)
        loss = args.criterion(outputs, targets)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            args.logger.info(progress.display(batch_idx))
    # TODO: this should also be done with the ProgressMeter
    args.logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
            .format(top1=top1, top5=top5))

    # Save checkpoint.
    acc = top1.avg
    args.logger.info('Epoch = {} : Acc = {}'.format(Epoch, acc))
    if (acc > args.best_acc) & (args.train == 0):
        args.logger.info('Saving model ...')
        state = {
            'net': args.net.state_dict(),
            'acc': acc,
            'epoch': Epoch + 1,
            'optimizer': args.optimizer.state_dict()
        }
        torch.save(state, args.save_path)
        args.best_acc = acc
        args.best_acc_epoch = Epoch
    elif (args.train == 1):
        args.logger.info("Testing  accuracy: {}".format(acc))

    # Record loss and accuracy from the test run into the writer
    args.writer.add_scalar('Test/Loss', losses.avg, Epoch)
    args.writer.add_scalar('Test/Acc', acc, Epoch)
    args.writer.flush()
    return top1.avg


