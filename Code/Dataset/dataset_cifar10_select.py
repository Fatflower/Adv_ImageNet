# %Header File Start-----------------------------------------------------------
#  Confidential（Unclassified）
#  COPYRIGHT (C) Sun Yat-sen University
#  THIS FILE MAY NOT BE MODIFIED OR REDISTRIBUTED WITHOUT THE
#  EXPRESSED WRITTEN CONSENT OF SYSU
# 
# %-----------------------------------------------------------------------------
#  Title   : dataset_cifar10_select.py
#  Author  : Zhang wentao; 
#  E-mail  : z1282429194@163.com
#  Created : 08/07/2021
#  Description: This part is a modification of the cifar10 dataset provided by pytorch, 
#               in order to arbitrarily select the class and control the number of 
#               samples in the selected class. This part of the code mainly refers to: 
#               https://github.com/Jianfei2333/pytorch-adversarial-training/blob/main/dataset.py
# 
# %-----------------------------------------------------------------------------
#  Modification History:
#  V1.0: 2021.09.11, first created by Zhang wentao 
#  V1.1: 2021.09.12, first modified by Zhang wentao
#
# %Header File End--------------------------------------------------------------


import glob
import os

import numpy as np
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms as T


class Cifar10(Dataset):
    def __init__(self, dataroot, transform=T.Compose([]), train=True, subset=10, max_n_per_class=10000):
        self.dataroot_param = dataroot
        self.dataroot = dataroot
        self.train = train
        # Class number of subset. Take top-N classes or same specific classes as a subset(Torchvision official implementation sorting).
        self.subset = subset

        self.max_n_per_class = max_n_per_class

        self.transform = transform

        self.data = datasets.CIFAR10(root=self.dataroot, train=train, transform=self.transform, download=True)

        # Metadata of dataset
        self.classes = self.data.classes
        self.class_num = len(self.data.classes)
        self.class_to_idx = self.data.class_to_idx
        self.idx_to_class = {self.class_to_idx[cls]:cls for cls in self.class_to_idx}

        # Subset process.
        if isinstance(self.subset, int):
            self.class_subset = list(range(self.subset))
        else:
            self.class_subset = list(self.subset)

        # self.mapping = {i: self.class_subset[i] for i in range(len(self.class_subset))}
        # self._rev_mapping = {self.mapping[i]: i for i in self.mapping}
        # self._rev_mapping = np.array([self._rev_mapping[i] if i in self.class_subset else -1 for i in range(10)])
        # target_mapping = lambda x: self._rev_mapping[x]

        self.subset_mask = np.array(self.data.targets)
        for i in self.class_subset:
            self.subset_mask[np.where(self.subset_mask == i)[0][self.max_n_per_class:]] = -1
        self.subset_indices = np.where(self.subset_mask != -1)[0]
        self.class_selection = np.where(np.in1d(np.array(self.data.targets), np.array(self.class_subset)) == 1)[0]
        self.subset_indices = np.intersect1d(self.subset_indices, self.class_selection)

        # Metadata override.
        self.classes = [self.classes[i] for i in self.class_subset]
        self.class_num = len(self.classes)
        self.idx_to_class = {i: self.idx_to_class[i] for i in self.class_subset}
        self.class_to_idx = {self.idx_to_class[i]: i for i in self.idx_to_class}

            
    
    def __getitem__(self, idx):
        idx_in_ori_data = self.subset_indices[idx]
        return self.data.__getitem__(idx_in_ori_data)

    def __len__(self):
        return len(self.subset_indices)

    def __repr__(self):
        repr = """Cifar10 Dataset(subset):
\tRoot location: {}
\tSplit: {}
\tClass num: {}
\tData num: {}""".format(self.dataroot, 'Train' if self.train else 'Test', self.class_num, self.__len__())
        return repr

if __name__=="__main__":
    # Use case
    data_path = '/home/disk2/hulai/Datasets/CIFAR10'
    # Set the first seven classes (0-4) as normal classes as the training set, and two samples per class.
    trainset = Cifar10(dataroot=data_path, train=True, subset=5, max_n_per_class=2)
    print(trainset)
    for i in range(len(trainset)):
        print(trainset[i][1])

    # Set some specific  classes ([0,3,6,7,9,1])  as normal classes as the training set, and three samples per class.
    trainset1 = Cifar10(dataroot=data_path, train=True, subset=[0,3,6,7,9,1], max_n_per_class=3)
    print(trainset1)
    for i in range(len(trainset1)):
        print(trainset1[i][1])
   
    