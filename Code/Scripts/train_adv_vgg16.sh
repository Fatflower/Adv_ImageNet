#!/bin/bash
cd ..
python ./Source/main_adv.py  --a 'vgg16' --classes_num 1000  --lr 1e-2 --trainbs 128  \
                              --testbs 128 --epoch 200 --gpu '3,5' --data_dir '/home/20/wentao/Datasets/ilsvrc2012' \
                              --info_log 'adv_ImageNet_'