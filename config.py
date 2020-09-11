#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/7/7 17:06
# @Author  : zhouyu
# @content : 
# @File    : config.py
# @Software: PyCharm

import data_handler

# gpu
device = 0

# modify this hyperparameter to get different binary code length
# code length = subspace_num * subspace_num * log2(subcenter_num) = 8*subspace_num
subspace_num = 4

# please modify these when change datasets
txt_dim = 1386
label_dim = 24
image_width = 256
image_height = 256
image_channel = 3

subcenter_num = 256
output_dim = 128

# triplet loss margin
margin = 0.5
# weights of loss function
beta = [0.3, 1.5, 0.01]

lr_vgg = 0.001
lr_dis = 0.001
lr_label = 0.001

# modify these two parameters following README file to get a better results for different datasets
lr_img = 0.06
lr_txt = 0.03

num_epochs_per_decay = 10
moving_average_decay = 0.9999
learning_rate_decay_factor = 0.5
max_iter_update_Cb = 2
max_iter_update_b = 2
train_num = data_handler.TRAINING_SIZE
test_num = data_handler.QUERY_SIZE
db_num = data_handler.DATABASE_SIZE
batch_size = 64
val_batch_size = 100
code_batch_size = 100
training_epoch = 30
# map@50
R = 50