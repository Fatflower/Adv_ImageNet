# %Header File Start-----------------------------------------------------------
#  Confidential（Unclassified）
#  COPYRIGHT (C) Sun Yat-sen University
#  THIS FILE MAY NOT BE MODIFIED OR REDISTRIBUTED WITHOUT THE
#  EXPRESSED WRITTEN CONSENT OF SYSU
# 
# %-----------------------------------------------------------------------------
#  Title   : transform_dataset.py
#  Author  : Zhang wentao; 
#  E-mail  : z1282429194@163.com
#  Created : 07/18/2021
#  Description: This file contains various transforms for different data sets. 
# 
# %-----------------------------------------------------------------------------
#  Modification History:
#  V1.0: 2021.08.08, first created by Zhang wentao
# 
# %Header File End--------------------------------------------------------------

import torchvision.transforms as transforms

# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]

# img_size = 256
# crop_size = 224

def common_train_transfoms(img_size, crop_size, mean, std):
    '''
    Image_size can be set to: (32,32), (96,96), (224,224)
    CIFAR10, CIFAR100, SVHN : (32,32)
    STL-10 : (96,96)
    ImageNet : (224,224)
    '''
    return transforms.Compose([
        # transforms.RandomCrop(image_size[0], padding=4),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation((0, 10)),
        transforms.RandomResizedCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
def common_test_transfoms(img_size, crop_size, mean, std):
    return transforms.Compose([
        # transforms.RandomResizedCrop(image_size),
        transforms.Resize(img_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

