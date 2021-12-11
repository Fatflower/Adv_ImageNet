# %Header File Start-----------------------------------------------------------
#  Confidential（Unclassified）
#  COPYRIGHT (C) Sun Yat-sen University
#  THIS FILE MAY NOT BE MODIFIED OR REDISTRIBUTED WITHOUT THE
#  EXPRESSED WRITTEN CONSENT OF SYSU
# 
# %-----------------------------------------------------------------------------
#  Title   : log.py
#  Author  : Zhang wentao; 
#  E-mail  : z1282429194@163.com
#  Created : 07/15/2021
#  Description: This part is to record the log information during training.
# 
# %-----------------------------------------------------------------------------
#  Modification History:
#  V1.0: 2021.07.15, first created by Zhang wentao
# 
# %-----------------------------------------------------------------------------


import logging
import time
import os

def log_creater(output_dir, info_log):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log_name = info_log + '_{}.log'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
    final_log_file = os.path.join(output_dir,log_name)


    # creat a log
    log = logging.getLogger('train_log')
    log.setLevel(logging.DEBUG)

    # FileHandler
    file = logging.FileHandler(final_log_file)
    file.setLevel(logging.DEBUG)

    # StreamHandler
    stream = logging.StreamHandler()
    stream.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter(
        '[%(asctime)s][line: %(lineno)d] =====> %(message)s')

    # setFormatter
    file.setFormatter(formatter)
    stream.setFormatter(formatter)

    # addHandler
    log.addHandler(file)
    log.addHandler(stream)

    log.info('creating {}'.format(final_log_file))
    return log


