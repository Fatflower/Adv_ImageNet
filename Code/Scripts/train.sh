#!/bin/bash
cd ..
python ./Source/main.py  --a 'resnet18' --classes_num 10  --lr 1e-2 --trainbs 128  \
                              --testbs 128 --epoch 90  --data_dir '/home/disk2/hulai/Datasets/CIFAR10' \
                              --info_log '_CIFAR10_'