#!/bin/bash
cd ..
python ./Source/tmp.py --a resnet50 --dist-url 'tcp://127.0.0.1:9999' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 '/home/disk2/hulai/Datasets/ilsvrc2012'