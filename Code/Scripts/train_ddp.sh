#!/bin/bash
cd ..
python ./Source/main_ddp.py  --a 'resnet34' --GPU '5,6,7' --info_log 'ImageNet' --dist-url 'tcp://127.0.0.1:23456' --dist-backend 'nccl'\
                             --multiprocessing-distributed --world-size 1 --rank 0 --data '/home/disk2/hulai/Datasets/ilsvrc2012'