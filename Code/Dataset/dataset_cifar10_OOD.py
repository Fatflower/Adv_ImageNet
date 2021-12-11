# %Header File Start-----------------------------------------------------------
#  Confidential（Unclassified）
#  COPYRIGHT (C) Sun Yat-sen University
#  THIS FILE MAY NOT BE MODIFIED OR REDISTRIBUTED WITHOUT THE
#  EXPRESSED WRITTEN CONSENT OF SYSU
# 
# %-----------------------------------------------------------------------------
#  Title   : dataset_cifar10_OOD.py
#  Author  : Zhang wentao; 
#  E-mail  : z1282429194@163.com
#  Created : 08/07/2021
#  Description: This part of the data set cifar10 is rewritten for the OOD task,
#               mainly refer to https://pytorch.org/vision/stable/_modules/torchvision/datasets/cifar.html 
# 
# %-----------------------------------------------------------------------------
#  Modification History:
#  V1.0: 2021.08.07, first created by Zhang wentao 
#
# %Header File End--------------------------------------------------------------


from PIL import Image
import os
import os.path
import numpy as np
import pickle
from typing import Any, Callable, Optional, Tuple
import torchvision

class train_CIFAR10(torchvision.datasets.CIFAR10):
     def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            num_up: int = 7,
    ) -> None:
        super(train_CIFAR10, self).__init__(root, transform=transform,
                                      target_transform=target_transform)
        self.train = train  # training set or test set
        if download:
                self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                                ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        tdata = []
        ttarget = []
        for (i, j) in zip(self.data, self.targets):
            if j < num_up:
                tdata.append(i)
                ttarget.append(j)
        self.data = tdata
        self.targets = ttarget
        self._load_meta()
    

class test_CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            num_up: int = 7,
    ) -> None:

        super(test_CIFAR10, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()
        self.num_up = num_up

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
            """
            Args:
                index (int): Index

            Returns:
                tuple: (image, target) where target is index of the target class.
            """
            img, target = self.data[index], self.targets[index]

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)
            target = self.num_up if target > self.num_up else target

            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, target


# 
if __name__=="__main__":
    data_path = '/home/disk2/hulai/Datasets/CIFAR10'
    # Set the first seven classes (0-6) as normal classes as the training set.
    trainset = train_CIFAR10(root=data_path, train=True, download=True, num_up=7)
    
    # Set the first seven classes (0-6) as normal classes, and the last three 
    # classes (7-9) as abnormal class to form a test set.
    testset = test_CIFAR10(root=data_path, train=False, download=True, num_up=7)
    for i in range(1000,10000,500):
        print(trainset[i][1], testset[i][1])