#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/7/8 15:38
# @Author  : zhouyu
# @content : 
# @File    : train_scripts.py
# @Software: PyCharm

from data_handler import *
import DAQN
import sys
import config
import time
# get the train data

a = DataSet()
if a.load is False:
    print("data error")
    sys.exit()
else:
    file_d = open('./result_data.txt', 'a')
    start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    file_d.write('\n======='+str(start_time)+'=======')
    model_dq = DAQN.train_test(a)
    file_d.write('\n====code length=' + str(config.subspace_num*8) +
                 "\nimg_lr:"+str(config.lr_img)+"-txt_lr:"+str(config.lr_txt)+"-label_lr:"+str(config.lr_label)
                 + "-discriminator_lr"+str(config.lr_dis)+"-beta_setting"+str(config.beta))
    file_d.write("\n")
    for k, v in model_dq.items():
        file_d.write(k+":")
        file_d.write(str(v) + '\n')
    file_d.close()
    sys.exit()
