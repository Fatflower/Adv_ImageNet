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


def test_refine(Epoch, args, dataloader):
    
    args.net.eval()
    test_loss = 0
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    # with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(dataloader):

        inputs, targets = inputs.to(args.device), targets.to(args.device)

        outputs = args.net(inputs)
        loss = args.criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        testing_acc  = (correct/total)
        testing_loss = (test_loss/(batch_idx+1))
        if batch_idx % args.print_freq == 0:
            args.logger.info('===== Test(clean) =====> Epoch: [{}/{}]     testing_loss = {:2.5f}    testing_acc = {:.5f}    testing_batchsize = {}'.format(Epoch, args.end_epoch-1, testing_loss,  testing_acc, (args.testbs))) # testing_loss=={:.5f}\t testing_acc={:.3f}
        
        y_true += (targets.cpu().numpy().tolist())
        y_pred += (predicted.cpu().numpy().tolist())
    args.logger.info(confusion_matrix(y_true, y_pred))
    

    # Save checkpoint.
    acc = 100.*correct/total
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
    args.writer.add_scalar('Test/Loss', test_loss, Epoch)
    args.writer.add_scalar('Test/Acc', acc, Epoch)
    args.writer.flush()


