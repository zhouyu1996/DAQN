#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/7/7 17:05
# @Author  : zhouyu
# @content : 
# @File    : DAQN.py
# @Software: PyCharm

import tensorflow as tf
import numpy as np
from utils.boundary_controlled_triplet_loss import *
from utils.dataset import *
from utils.calc_map import *
import config
from bone_net import *
import random
import time
from datetime import datetime
from sklearn.cluster import MiniBatchKMeans

class DAQN(object):
    def __init__(self):
        with tf.device(config.device):
            configProt = tf.ConfigProto()
            configProt.gpu_options.allow_growth = True
            configProt.allow_soft_placement = True
            self.sess = tf.Session(config=configProt)
        # placeholder
        with tf.device(config.device):
            self.C = tf.Variable(tf.random_uniform([config.subspace_num * config.subcenter_num, config.output_dim],
                                                   minval=-1, maxval=1, dtype=tf.float32, name='codebook'))
            self.l = tf.placeholder(tf.float32, [])
            self.train_flag = tf.placeholder(tf.bool)
            self.imgs = tf.placeholder(tf.float32,
                                       [None, config.image_height, config.image_width,config.image_channel])
            self.img_label = tf.placeholder(tf.float32, [None, config.label_dim])
            self.txt_label = tf.placeholder(tf.float32, [None, config.label_dim])
            self.txts = tf.placeholder(tf.float32, [None, config.txt_dim])
            self.subspace_num = config.subspace_num
            self.subcenter_num = config.subcenter_num
            self.b_img = tf.placeholder(tf.float32, [None, config.subspace_num * config.subcenter_num])
            self.b_txt = tf.placeholder(tf.float32, [None, config.subspace_num * config.subcenter_num])
            self.ICM_m = tf.placeholder(tf.int32, [])
            self.ICM_b_m = tf.placeholder(tf.float32, [None, config.subcenter_num])
            self.ICM_b_all = tf.placeholder(tf.float32, [None, config.subcenter_num * config.subspace_num])
            self.ICM_X = tf.placeholder(tf.float32, [config.code_batch_size, config.output_dim])
            self.ICM_C_m = tf.slice(self.C, [self.ICM_m * self.subcenter_num, 0],
                                    [self.subcenter_num, config.output_dim])
            self.ICM_X_residual = tf.add(tf.subtract(self.ICM_X, tf.matmul(self.ICM_b_all, self.C)),
                                         tf.matmul(self.ICM_b_m, self.ICM_C_m))
            ICM_X_expand = tf.expand_dims(self.ICM_X_residual, 2)
            ICM_C_m_expand = tf.expand_dims(tf.transpose(self.ICM_C_m), 0)
            ICM_sum_squares = tf.reduce_sum(tf.square(tf.squeeze(tf.subtract(ICM_X_expand, ICM_C_m_expand))),
                                            reduction_indices=1)
            ICM_best_centers = tf.argmin(ICM_sum_squares, 1)
            self.ICM_best_centers_one_hot = tf.one_hot(ICM_best_centers, config.subcenter_num, dtype=tf.float32)

        with tf.device(config.device):
            self.img_output_all = tf.placeholder(tf.float32, [None, config.output_dim])
            self.txt_output_all = tf.placeholder(tf.float32, [None, config.output_dim])
            self.img_b_all = tf.placeholder(tf.float32, [None, config.subspace_num * config.subcenter_num])
            self.txt_b_all = tf.placeholder(tf.float32, [None, config.subspace_num * config.subcenter_num])
            # network structure
            self.img_net = img_net
            self.txt_net = txt_net
            self.label_classifier = label_classifier
            self.domain_classifier = domain_classifier

        self.build_model()

    def build_model(self):

        self.img_featrue = self.img_net(self.imgs, config.output_dim, self.train_flag,reuse=False)
        self.txt_featrue = self.txt_net(self.txts, config.output_dim, self.train_flag,reuse=False)

        self.logits_v = self.label_classifier(self.img_featrue)
        self.logits_w = self.label_classifier(self.txt_featrue, reuse=True)

        self.emb_v_class = self.domain_classifier(self.img_featrue, self.l)
        self.emb_w_class = self.domain_classifier(self.txt_featrue, self.l, reuse=True)

        # classification loss
        self.label_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.img_label, logits=self.logits_v) + \
                          tf.nn.softmax_cross_entropy_with_logits(labels=self.txt_label, logits=self.logits_w)
        self.label_loss = config.beta[0] * tf.reduce_mean(self.label_loss)

        # adversarial loss
        # label smooth
        smooth = 0.1
        all_emb_v = tf.concat([tf.ones([config.batch_size, 1]),
                               tf.zeros([config.batch_size, 1])], 1)
        all_emb_v = all_emb_v*(1.0-smooth)
        all_emb_w = tf.concat([tf.zeros([config.batch_size, 1]),
                               tf.ones([config.batch_size, 1])], 1)
        all_emb_w = all_emb_w*(1.0-smooth)
        self.domain_class_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.emb_v_class,
                                                                         labels=all_emb_w) + \
                                 tf.nn.softmax_cross_entropy_with_logits(logits=self.emb_w_class, labels=all_emb_v)
        self.domain_class_loss = tf.reduce_mean(self.domain_class_loss)

        # boundary controlled triplet loss
        self.triplet_loss1 = triplet_semihard_loss(self.img_featrue, self.txt_featrue,self.img_label,
                                                     self.txt_label,config.margin)
        self.triplet_loss2 = triplet_semihard_loss(self.txt_featrue, self.img_featrue, self.txt_label,
                                                   self.img_label,config.margin)
        self.triplet_loss = config.beta[1] * tf.add(self.triplet_loss1, self.triplet_loss2)

        # quantization loss
        self.cq_loss_img = tf.reduce_mean(
            tf.reduce_sum(tf.square(tf.subtract(self.img_featrue, tf.matmul(self.b_img, self.C))), 1))
        self.cq_loss_txt = tf.reduce_mean(
            tf.reduce_sum(tf.square(tf.subtract(self.txt_featrue, tf.matmul(self.b_txt, self.C))), 1))
        self.q_lambda = tf.Variable(config.beta[2], name='lambda')
        self.cq_loss = tf.multiply(self.q_lambda, tf.add(self.cq_loss_img, self.cq_loss_txt))

        # total loss
        self.total_loss = self.label_loss + self.triplet_loss + self.cq_loss + self.domain_class_loss

    def initial_centers(self, img_output, txt_output):
        C_init = np.zeros([self.subspace_num * self.subcenter_num, config.output_dim])
        print("===initilizing C===")
        all_output = np.vstack([img_output, txt_output])
        for i in range(self.subspace_num):
            kmeans = MiniBatchKMeans(n_clusters=self.subcenter_num).fit(
                all_output[:,
                i * config.output_dim // self.subspace_num: (i + 1) * config.output_dim // self.subspace_num])
            C_init[i * self.subcenter_num: (i + 1) * self.subcenter_num, i * config.output_dim // self.subspace_num: (
                i + 1) * config.output_dim // self.subspace_num] = kmeans.cluster_centers_

        return C_init

    # these codes comes from CAOYUE 2017-AAAI-CDQ
    def update_centers(self, img_dataset, txt_dataset):
        h = tf.concat([self.img_b_all, self.txt_b_all], 0)
        U = tf.concat([self.img_output_all, self.txt_output_all], 0)
        smallResidual = tf.constant(np.eye(self.subcenter_num * self.subspace_num, dtype=np.float32) * 0.001)
        Uh = tf.matmul(tf.transpose(h), U)
        hh = tf.add(tf.matmul(tf.transpose(h), h), smallResidual)
        compute_centers = tf.matmul(tf.matrix_inverse(hh), Uh)
        update_C = self.C.assign(compute_centers)
        C_value = self.sess.run(update_C, feed_dict={
            self.img_output_all: img_dataset.full_output_without_shuffle(),
            self.txt_output_all: txt_dataset.full_output_without_shuffle(),
            self.img_b_all: img_dataset.full_codes_without_shuffle(),
            self.txt_b_all: txt_dataset.full_codes_without_shuffle(),
        })
        print('===update C===')
        print('the number of nonzero elements of codebook is:',len(np.where(np.sum(C_value, 1) != 0)[0]))

    def update_codes_ICM(self, output, code):
        code = np.zeros(code.shape)
        for iterate in range(config.max_iter_update_b):
            sub_list = [i for i in range(self.subspace_num)]
            random.shuffle(sub_list)
            for m in sub_list:
                best_centers_one_hot_val = self.sess.run(self.ICM_best_centers_one_hot, feed_dict={
                    self.ICM_b_m: code[:, m * self.subcenter_num: (m + 1) * self.subcenter_num],
                    self.ICM_b_all: code,
                    self.ICM_m: m,
                    self.ICM_X: output,
                })
                code[:, m * self.subcenter_num: (m + 1) * self.subcenter_num] = best_centers_one_hot_val
        return code

    def update_codes_batch(self, dataset, batch_size):
        total_batch = int(dataset.n() / batch_size)
        print("===update B in batchsize:", batch_size,"===")
        # shuffle order of dataset
        dataset.finish_epoch()
        for i in range(total_batch):
            output_val, code_val = dataset.next_batch_output_codes(batch_size)
            codes_val = self.update_codes_ICM(output_val, code_val)
            dataset.copy_batch_codes(codes_val, batch_size)
        print("===update_code wrong:", np.sum(np.sum(dataset.codes, 1) != config.subspace_num))

    def train_deep_networks(self, global_step):
        num_batches_per_epoch = int(config.train_num / config.batch_size)
        decay_steps = int(num_batches_per_epoch * config.num_epochs_per_decay)
        self.lr_vgg = tf.train.exponential_decay(config.lr_vgg, global_step, decay_steps,
                                                 config.learning_rate_decay_factor, staircase=True)
        self.img_lr = tf.train.exponential_decay(config.lr_img, global_step, decay_steps,
                                                 config.learning_rate_decay_factor, staircase=True)
        self.txt_lr = tf.train.exponential_decay(config.lr_txt, global_step, decay_steps,
                                                 config.learning_rate_decay_factor, staircase=True)
        self.label_lr = tf.train.exponential_decay(config.lr_label, global_step, decay_steps,
                                                 config.learning_rate_decay_factor, staircase=True)
        self.dom_lr = tf.train.exponential_decay(config.lr_dis, global_step, decay_steps,
                                                      config.learning_rate_decay_factor, staircase=True)
        self.t_vars = tf.trainable_variables()
        self.vgg_vars = [v for v in self.t_vars if 'conv' or 'fc' in v.name]
        self.dc_vars  = [v for v in self.t_vars if 'dc_' in v.name]
        self.lc_vars  = [v for v in self.t_vars if 'lc_' in v.name]
        self.img_vars = [v for v in self.t_vars if 'img' in v.name]
        self.txt_vars = [v for v in self.t_vars if 'txt' in v.name]

        # BN layer
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            apply_gradient_op_vgg = tf.train.AdamOptimizer(
                learning_rate=self.lr_vgg, beta1=0.5).minimize(self.total_loss,var_list = self.vgg_vars)
            apply_gradient_op_img = tf.train.AdamOptimizer(learning_rate=self.img_lr, beta1=0.5).minimize(
                self.total_loss, var_list=self.img_vars, global_step=global_step)
            apply_gradient_op_txt = tf.train.AdamOptimizer(learning_rate=self.txt_lr, beta1=0.5).minimize(
                self.total_loss, var_list=self.txt_vars, global_step=global_step)
            apply_gradient_op_label = tf.train.AdamOptimizer(learning_rate=self.label_lr, beta1=0.5).minimize(
                self.total_loss, var_list=self.lc_vars, global_step=global_step)
            apply_gradient_op_dis = tf.train.AdamOptimizer(learning_rate=self.dom_lr, beta1=0.5).minimize(
                self.domain_class_loss, var_list=self.dc_vars, global_step=global_step,)

        apply_gradient_op_gen = tf.group(apply_gradient_op_img,apply_gradient_op_txt,
                                         apply_gradient_op_label,apply_gradient_op_vgg)
        return apply_gradient_op_gen, apply_gradient_op_dis

    def train(self, img_dataset, txt_dataset):
        with tf.device(config.device):
            global_step = tf.Variable(0, trainable=False)
            self.global_step = global_step
            gen_gradient_op, dis_gradient_op = self.train_deep_networks(global_step)
            init = tf.global_variables_initializer()
            self.sess.run(init)

            print("=== DAQN Training procedure start ===")
        global_step = 0
        for epoch in range(config.training_epoch):
            p = float(epoch) / config.training_epoch
            l = 2. / (1. + np.exp(-10. * p)) - 1
            step = 0
            total_batch = int(config.train_num / config.batch_size)
            print(" Epoch: " + str(epoch))
            print("======DAQN Train# training total batch is " + str(total_batch))
            if epoch > 0:
                # alternative training
                with tf.device(config.device):
                    for i in range(config.max_iter_update_Cb):
                        print("===DAQN Train: update B and C in ", i, " iter===")
                        if epoch == 1:
                            self.sess.run(self.C.assign(self.initial_centers(img_dataset.full_output_without_shuffle(),
                                                                             txt_dataset.full_output_without_shuffle())))

                        self.update_codes_batch(img_dataset, config.code_batch_size)
                        self.update_codes_batch(txt_dataset, config.code_batch_size)
                        self.update_centers(img_dataset, txt_dataset)
                    print("===update C and B done!===")

            with tf.device(config.device):
                for i in range(total_batch):
                    images, image_labels, image_codes = img_dataset.next_batch(config.batch_size)
                    texts, text_labels, text_codes = txt_dataset.next_batch(config.batch_size)
                    if epoch > 0:
                        assign_lambda = self.q_lambda.assign(config.beta[2])
                    else:
                        assign_lambda = self.q_lambda.assign(0.0)
                    self.sess.run([assign_lambda])
                    _, total_loss, label_loss, domain_class_loss, cq_loss, triplet_loss, \
                    batch_img_output, batch_txt_output \
                        = self.sess.run([gen_gradient_op, self.total_loss, self.label_loss, self.domain_class_loss,
                                         self.cq_loss, self.triplet_loss, self.img_featrue, self.txt_featrue, ],
                                        feed_dict={self.imgs: images, self.img_label: image_labels,
                                                   self.b_img: image_codes,
                                                   self.txts: texts, self.txt_label: text_labels,
                                                   self.b_txt: text_codes,
                                                   self.l: l, self.train_flag: True})
                    print('===generator training===')
                    format_str = (
                        '%s: step %4d,label_loss = %.4f,domain_class_loss = %.4f,triplet_loss=%.4f,cq loss = %.4f ')
                    print(format_str % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), global_step,
                                        label_loss, domain_class_loss,triplet_loss, cq_loss))
                    print("G====total loss===", total_loss)

                    if(global_step % 10 ==0):
                        _, total_loss, label_loss, domain_class_loss, cq_loss, triplet_loss,\
                        batch_img_output, batch_txt_output \
                            = self.sess.run([dis_gradient_op, self.total_loss, self.label_loss, self.domain_class_loss,
                                             self.cq_loss, self.triplet_loss, self.img_featrue, self.txt_featrue,],
                                            feed_dict={self.imgs: images, self.img_label: image_labels,
                                                       self.b_img: image_codes,
                                                       self.txts: texts, self.txt_label: text_labels,
                                                       self.b_txt: text_codes,
                                                       self.l: l, self.train_flag: True})
                        print('===modality discriminator training===')
                        format_str = (
                            '%s: step %4d,label_loss = %.4f,domain_class_loss = %.4f,triplet_loss=%.4f,cq loss = %.4f ')
                        print(
                            format_str % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), global_step,
                                          label_loss, domain_class_loss, triplet_loss, cq_loss))
                        print("G====total loss===", total_loss)
                    # update featrues
                    img_dataset.copy_batch_output(batch_img_output, config.batch_size)
                    txt_dataset.copy_batch_output(batch_txt_output, config.batch_size)
                    # NAN assert
                    assert not np.isnan(label_loss)
                    assert not np.isnan(domain_class_loss)
                    assert not np.isnan(triplet_loss)
                    assert not np.isnan(cq_loss)
                    step += 1
                    global_step += 1

    def validation(self, database_img, database_txt, query_img, query_txt):
        with tf.device(config.device):
            print("===DADN Validation===")

            total_batch = int(config.db_num / config.val_batch_size)
            print("===map dataset point to common subspace with trained DAQN===")
            for i in range(total_batch):
                images, image_labels, image_codes = database_img.next_batch(config.val_batch_size)
                texts, text_labels, text_codes = database_txt.next_batch(config.val_batch_size)
                start_time = time.time()
                batch_img_output, batch_txt_output = self.sess.run(
                    [self.img_featrue, self.txt_featrue],
                    feed_dict={self.imgs: images, self.txts: texts, self.img_label: image_labels,
                               self.txt_label: text_labels, self.train_flag: False})
                database_img.copy_batch_output(batch_img_output, config.val_batch_size)
                database_txt.copy_batch_output(batch_txt_output, config.val_batch_size)
                duration = time.time() - start_time
                # ==================
                print(str(i) + " / " + str(total_batch) + " batch, time=" + str(duration))
                # OOM
                del images, texts
            # update codes of database
            self.update_codes_batch(database_img, config.code_batch_size)
            self.update_codes_batch(database_txt, config.code_batch_size)

            total_batch = int(config.test_num / config.val_batch_size)
            print("===map query point to common subspace with trained DAQN===")
            for i in range(total_batch):
                images, image_labels, image_codes = query_img.next_batch(config.val_batch_size)
                texts, text_labels, text_codes = query_txt.next_batch(config.val_batch_size)
                batch_img_output, batch_txt_output = self.sess.run(
                    [self.img_featrue, self.txt_featrue],
                    feed_dict={self.imgs: images, self.txts: texts, self.img_label: image_labels,
                               self.txt_label: text_labels,self.train_flag:False})
                query_img.copy_batch_output(batch_img_output, config.val_batch_size)
                query_txt.copy_batch_output(batch_txt_output, config.val_batch_size)
                # OOM
                del images, texts
            # update codes of query
            self.update_codes_batch(query_img, config.code_batch_size)
            self.update_codes_batch(query_txt, config.code_batch_size)

            print("===compute MAP score with Asymmetric Quantizer Distance AQD===")
            mAPs = MAPs(self.sess.run(self.C), self.subspace_num, self.subcenter_num, config.R)
            return {
                'i2t_AQD': mAPs.get_mAPs_AQD(database_txt, query_img),
                't2i_AQD': mAPs.get_mAPs_AQD(database_img, query_txt),
            }

def train_test(a):
    model = DAQN()
    train_data = a.import_train()
    img_dataset = Dataset(train_data[1],train_data[3], config.subspace_num * config.subcenter_num, config.output_dim)
    txt_dataset = Dataset(train_data[2],train_data[3], config.subspace_num * config.subcenter_num, config.output_dim)
    model.train(img_dataset, txt_dataset)
    del train_data
    query_data = a.test_data()
    db_data = a.db_data()
    print('======test process start=====')
    database_img = Dataset(db_data[1], db_data[3], config.subspace_num * config.subcenter_num, config.output_dim)
    database_txt = Dataset(db_data[2], db_data[3], config.subspace_num * config.subcenter_num, config.output_dim)
    query_img = Dataset(query_data[1], query_data[3], config.subspace_num * config.subcenter_num, config.output_dim)
    query_txt = Dataset(query_data[2], query_data[3], config.subspace_num * config.subcenter_num, config.output_dim)
    ret = model.validation(database_img, database_txt, query_img, query_txt)
    print(ret)
    del model
    return ret
