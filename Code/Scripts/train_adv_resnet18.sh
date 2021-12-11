#!/bin/bash
cd ..
python ./Source/main_adv.py  --a 'resnet18' --classes_num 1000  --lr 1e-3 --trainbs 256  --resume\
                              --testbs 256 --epoch 150 --gpu '4' --data_dir '/home/20/wentao/Datasets/ilsvrc2012' \
                              --info_log 'adv_ImageNet_'
