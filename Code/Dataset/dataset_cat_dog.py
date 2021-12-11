# %Header File Start-----------------------------------------------------------
#  Confidential（Unclassified）
#  COPYRIGHT (C) Sun Yat-sen University
#  THIS FILE MAY NOT BE MODIFIED OR REDISTRIBUTED WITHOUT THE
#  EXPRESSED WRITTEN CONSENT OF SYSU
# 
# %-----------------------------------------------------------------------------
#  Title   : dataset_cat_dog.py
#  Author  : Zhang wentao; 
#  E-mail  : z1282429194@163.com
#  Created : 07/15/2021
#  Description: This part is to load and preprocess DogvsCat dataset. 
# 
# %-----------------------------------------------------------------------------
#  Modification History:
#  V1.0: 2021.07.15, first created by Zhang wentao
# 
# %Header File End--------------------------------------------------------------
import  os
from PIL import Image
import glob
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


class DogvsCat(Dataset):
    def __init__(self, data_path, transform=True):
        '''
        data_path: The path of the dataset
        transform: Data preprocessing
        '''
        self.data_path = data_path
        self.transform_flag = transform


        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),            
            ])

        if self.transform_flag:
            self.transform = transform




        # load raw data
        classes_name = {
            'dogs' : 0,
            'cats' : 1
        }

        img_data_path = []
        label = []

        for item in classes_name:
            path = os.path.join(self.data_path, item)

            for impath in glob.glob(os.path.join(path, '*.jpg')):
                img_data_path.append(impath)
                label.append(classes_name[item])

        self.img_data_path = img_data_path
        self.label = label


    def __len__(self):
        return len(self.label)
    
    def __getitem__(self,index):
        img_path = self.img_data_path[index]
        img_label = self.label[index]
        img = Image.open(img_path)
        if self.transform_flag is not None:
            img = self.transform(img)
        return img, img_label

if __name__ == "__main__":
    data_path = '/home/disk2/hulai/Datasets/cat-vs-dog/test_set'
    dataset = DogvsCat(data_path)
    # img, img_label = dataset.__getitem__
    print(dataset[0][0].shape)
 
    
