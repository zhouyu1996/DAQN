#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/7/13 21:15
# @Author  : zhouyu
# @content : 
# @File    : bone_net.py
# @Software: PyCharm

import tensorflow.contrib.slim as slim
import numpy as np
import time
from utils.flip_gradient import *
import config

# VGGnet url
path = './models/vgg19.npy'
VGG_MEAN = [103.939, 116.779, 123.68]

def img_net(images, dim, train,reuse=False):
    # randomly resize images to 224*224*3
    distorted_image = tf.cond(tf.equal(train, tf.constant(True)),
                              lambda: tf.random_crop(images, [config.batch_size, 224, 224, 3]),
                              lambda: tf.random_crop(images, [config.val_batch_size, 224, 224, 3]))
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
    image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
    vgg = Vgg19()
    vgg.build(image)
    out = vgg.fc7
    features = out
    with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse,
                        weights_initializer=tf.truncated_normal_initializer(
                            stddev=0.01),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={'is_training': train, 'decay': 0.95}
                        ):
        net = tf.nn.relu(slim.fully_connected(features, 2048, scope='img_fc0'))
        net = tf.layers.dropout(net, rate=0.5, training=train)
        net = tf.nn.relu(slim.fully_connected(net, 1024, scope='img_fc1'))
        net = tf.layers.dropout(net, rate=0.5, training=train)
        net = slim.fully_connected(net, dim, scope='img_fc2')
    return tf.nn.tanh(net)

def txt_net(txts, dim, train,reuse=False):
    features = txts
    with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse,
                        weights_initializer=tf.truncated_normal_initializer(
                            stddev=0.01),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={'is_training': train, 'decay': 0.95},

    ):
        net = tf.nn.relu(slim.fully_connected(features, int(config.txt_dim / 2), scope='txt_fc0'))
        net = tf.layers.dropout(net, rate=0.5, training=train)
        net = tf.nn.relu(slim.fully_connected(net, int(config.txt_dim / 4), scope='txt_fc1'))
        net = tf.layers.dropout(net, rate=0.5, training=train)
        net = slim.fully_connected(net, dim, scope='txt_fc2')
    return tf.nn.tanh(net)

def label_classifier(emdedding,train=True, reuse=False):
    with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse,
                        weights_initializer=tf.truncated_normal_initializer(
                            stddev=0.01),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={'is_training': train, 'decay': 0.95},
    ):
        net = tf.nn.relu(slim.fully_connected(emdedding, int(config.output_dim / 2), scope='lc_fc0'))
        net = tf.layers.dropout(net, rate=0.5, training=True)
        net = slim.fully_connected(net, config.label_dim, scope='lc_fc1')
    return net

def domain_classifier(emdedding,l,train=True, reuse=False):
    with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse,
                        weights_initializer=tf.truncated_normal_initializer(
                            stddev=0.01),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={'is_training': train, 'decay': 0.95}
                        ):
        # GRL
        E = GRL(emdedding, l)
        net = tf.nn.relu(slim.fully_connected(E, int(config.output_dim / 2), scope='dc_fc0'))
        net = tf.layers.dropout(net, rate=0.5, training=True)
        net = slim.fully_connected(net, 2, scope='dc_fc1')
    return net

# this part codes come form https://github.com/2281123066/vgg-tensorflow
class Vgg19:
    def __init__(self, vgg19_npy_path=None):
        if vgg19_npy_path is None:
            vgg19_npy_path = path
        self.data_dict = np.load(vgg19_npy_path,allow_pickle=True, encoding='latin1').item()

    def build(self, rgb):
        start_time = time.time()
        # Convert RGB to BGR
        red, green, blue = tf.split(value=rgb, num_or_size_splits=3, axis=3)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.conv3_4 = self.conv_layer(self.conv3_3, "conv3_4")
        self.pool3 = self.max_pool(self.conv3_4, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.conv4_4 = self.conv_layer(self.conv4_3, "conv4_4")
        self.pool4 = self.max_pool(self.conv4_4, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.conv5_4 = self.conv_layer(self.conv5_3, "conv5_4")
        self.pool5 = self.max_pool(self.conv5_4, 'pool5')

        self.fc6 = self.fc_layer(self.pool5, "fc6")
        assert self.fc6.get_shape().as_list()[1:] == [4096]
        self.relu6 = tf.nn.relu(self.fc6)

        self.fc7 = self.fc_layer(self.relu6, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)

        self.fc8 = self.fc_layer(self.relu7, "fc8")

        self.prob = tf.nn.softmax(self.fc8, name="prob")

        self.data_dict = None
        print("build model finished: %ds" % (time.time() - start_time))

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")


if __name__ == '__main__':
    NET = Vgg19()
