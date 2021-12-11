# %Header File Start-----------------------------------------------------------
#  Confidential（Unclassified）
#  COPYRIGHT (C) Sun Yat-sen University
#  THIS FILE MAY NOT BE MODIFIED OR REDISTRIBUTED WITHOUT THE
#  EXPRESSED WRITTEN CONSENT OF SYSU
# 
# %-----------------------------------------------------------------------------
#  Title   : tool_kits.py
#  Author  : Zhang wentao; 
#  E-mail  : z1282429194@163.com
#  Created : 08/08/2021
#  Description: This is a commonly used script code library. 
# 
# %-----------------------------------------------------------------------------
#  Modification History:
#  V1.0: 2021.08.08, first created by Zhang wentao
# 
# %Header File End--------------------------------------------------------------

import os
import numpy as np
from numpy import linalg as LA
from scipy.spatial.distance import cdist
import torch.nn.functional as F
import torch

EPSILON = 1e-8

def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def distance_diff(x1, x2, metric):
    '''
    x1, x2: Their data type is a 2-dimensional array.
    metric: It is to choose which required distance. 
            It can be set as: cosine(余弦夹角), euclidean(欧式距离), 
            seuclidean(标准化欧式距离), sqeuclidean(平方欧几里得距离),
            hamming(汉明距离), etc. 
            Its type is string.
            For more specific details, please refer to: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
    '''
    y = cdist(x1, x2, metric=metric)
    return y

def norm_2(x1, x2):
    '''
    This is used to canculate 2_norm of two arraies. 
    '''
    y = x1 - x2
    out = LA.norm(y)
    return out

def uniform_vectors(x):
    # x = (x.T / (LA.norm(x.T, axis=0) + EPSILON)).T
    '''
    对二维数组按行归一化, p=2表示二范式, dim=1表示按行归一化
    '''
    x = F.normalize(x, p=2, dim=1)
    return x

#计算features与各类类中心之间的距离
def calculate_cos(features, class_means):
    c = class_means.size(0)
    b = features.size(0)
    cos_simi = torch.cosine_similarity(features.unsqueeze(dim=1).repeat(1,c,1),class_means.unsqueeze(dim=0).repeat(b,1,1),dim=2)
    # cos_dis = 1-cos_simi
    return cos_simi

def cal_dis(class_means, features):
    dis = torch.cdist(features.float(), class_means.float(), p=2)
    return dis

def cal_predicted_cos(dis_cos_class, cos_threshold, num, predicted):
    max_dis_cos_class = torch.max(dis_cos_class,dim=1)
    Val_max_dis_cos_class = max_dis_cos_class[0]
    idx_dis_cos_class = torch.lt(Val_max_dis_cos_class,cos_threshold)  #0.9 is cosine_similarity threshold, low 0.9 is abnormal
    ab_dis_cos_class = torch.ones_like(idx_dis_cos_class)*num
    predicted_cos = torch.where(idx_dis_cos_class ==True, ab_dis_cos_class, predicted)
    return predicted_cos

def cal_predicted_dis(dis_Eud_class, dis_threshold, num, predicted):
    min_dis_Eud_class = torch.min(dis_Eud_class,dim=1)
    Val_min_dis_Eud_class = min_dis_Eud_class[0]
    idx_dis_Eud_class = torch.gt(Val_min_dis_Eud_class, dis_threshold)  #0.5 is euclidean distance threshold, great 0.5 is abnormal
    ab_dis_Eud_class = torch.ones_like(idx_dis_Eud_class)*num
    predicted_dis = torch.where(idx_dis_Eud_class ==True, ab_dis_Eud_class, predicted)
    return predicted_dis

# Record all hyperparameters and paths used 
def record_para(args):
    args.logger.info('Model architecture : ' + args.a)
    args.logger.info('Number of classes  : {}' .format(args.classes_num))
    args.logger.info('Learning rate      : {}' .format(args.lr))
    args.logger.info('Train(0) or Test(1): {}' .format(args.train))
    args.logger.info('Batchsize during training: {}' .format(args.trainbs))
    args.logger.info('Batchsize during testing : {}' .format(args.testbs))
    args.logger.info('Epoch of model training  : {}' .format(args.epoch))
    args.logger.info('Print frequency          : {}' .format(args.print_freq))
    args.logger.info('Pretrained               : ' + ('false' if args.pretrained == False else 'True'))
    args.logger.info('Resume from checkpoint   : ' + ('false' if args.resume == False else 'True'))
    args.logger.info('Cuda_visible_devices     : ' + args.gpu)
    args.logger.info('Path of dataset          : ' + args.data_dir)
    args.logger.info('Seed for initializing training         : {}' .format(args.seed))
    args.logger.info('The mean used in data normalization    : {}' .format(args.mean))
    args.logger.info('The variance used in data normalization: {}' .format(args.std))