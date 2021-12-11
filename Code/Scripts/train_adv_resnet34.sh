#!/bin/bash
cd ..
python ./Source/main_adv.py  --a 'resnet34' --classes_num 1000  --lr 1e-2 --trainbs 256  \
                              --testbs 256 --epoch 200 --gpu '0,1' --data_dir '/home/20/wentao/Datasets/ilsvrc2012' \
                              --info_log 'adv_ImageNet_'