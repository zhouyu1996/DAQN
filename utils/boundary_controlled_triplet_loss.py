#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/7/18 15:14
# @Author  : zhouyu
# @content :
# @File    : boundary_controlled_triplet_loss.py
# @Software: PyCharm

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
import tensorflow as tf
import numpy as np
import config

def pairwise_distances(embeddings1,embeddings2):
    X1_X2 = tf.matmul(embeddings1, tf.transpose(embeddings2))
    inner_dis = tf.div(X1_X2, config.output_dim)
    sim = tf.div(inner_dis, 2) + 0.5
    return sim

def masked_maximum(data, mask, dim=1):
  axis_minimums = math_ops.reduce_min(data, dim, keep_dims=True)
  masked_maximums = math_ops.reduce_max(
      math_ops.multiply(data - axis_minimums, mask), dim,
      keep_dims=True) + axis_minimums
  return masked_maximums

def masked_minimum(data, mask, dim=1):
  axis_maximums = math_ops.reduce_max(data, dim, keep_dims=True)
  masked_minimums = math_ops.reduce_min(
      math_ops.multiply(data - axis_maximums, mask), dim,
      keep_dims=True) + axis_maximums
  return masked_minimums

def _get_anchor_positive_triplet_mask(img_labels,txt_labels):
    indices_not_equal = tf.cast(tf.ones([tf.shape(img_labels)[0], tf.shape(img_labels)[0]]), tf.bool)
    sim = tf.matmul(img_labels, tf.transpose(txt_labels))
    b = tf.zeros_like(sim, dtype=tf.int32)
    label_equal = tf.greater(tf.to_int32(sim), b)
    mask = tf.logical_and(indices_not_equal, label_equal)
    return mask

def _get_anchor_negative_triplet_mask(img_labels, txt_labels):
    sim = tf.matmul(img_labels, tf.transpose(txt_labels))
    b = tf.zeros_like(sim, dtype=tf.int32)
    label_equal = tf.greater(tf.to_int32(sim), b)
    mask = tf.logical_not(label_equal)
    return mask

# boundary controlled triplet loss with semi-hard sampling
# the code is based on tf.contrib.losses.metric_learning.triplet_semihard_loss
def triplet_semihard_loss(embeddings1,embeddings2,img_labels,txt_labels,margin=0.4):
    pdist_matrix = pairwise_distances(embeddings1, embeddings2)
    adjacency = _get_anchor_positive_triplet_mask(img_labels, txt_labels)
    adjacency_not = _get_anchor_negative_triplet_mask(img_labels,txt_labels)
    batch_size = tf.shape(img_labels)[0]
    pdist_matrix_tile = array_ops.tile(pdist_matrix, [batch_size, 1])
    mask = math_ops.logical_and(
      array_ops.tile(adjacency_not, [batch_size, 1]),
      math_ops.less(
          pdist_matrix_tile, array_ops.reshape(
              array_ops.transpose(pdist_matrix), [-1, 1])))
    mask_final = array_ops.reshape(
      math_ops.greater(
          math_ops.reduce_sum(
              math_ops.cast(mask, dtype=dtypes.float32), 1, keep_dims=True),
          0.0), tf.convert_to_tensor([batch_size, batch_size]))
    mask_final = array_ops.transpose(mask_final)
    adjacency_not = math_ops.cast(adjacency_not, dtype=dtypes.float32)
    mask = math_ops.cast(mask, dtype=dtypes.float32)
    # simahard
    negatives_outside = array_ops.reshape(
      masked_maximum(pdist_matrix_tile, mask), [batch_size, batch_size])
    negatives_outside = array_ops.transpose(negatives_outside)
    negatives_inside = array_ops.tile(
      masked_minimum(pdist_matrix, adjacency_not), [1, batch_size])
    semi_hard_negatives = array_ops.where(
      mask_final, negatives_outside, negatives_inside)
    loss_mat = math_ops.add(margin, -pdist_matrix + semi_hard_negatives)
    mask_positives = math_ops.cast(adjacency, dtype=dtypes.float32)
    num_positives = math_ops.reduce_sum(mask_positives)
    triplet_loss = math_ops.truediv(
        math_ops.reduce_sum(
            math_ops.maximum(
                math_ops.multiply(loss_mat, mask_positives), 0.0)),
    num_positives, name='triplet_semihard_loss')

    margin_p = config.margin_p
    margin_n = config.margin_n
    loss_mat1 = math_ops.add(margin_p, -pdist_matrix)
    loss_mat2 = math_ops.add(-margin_n, semi_hard_negatives)
    triplet_loss1 = math_ops.truediv(
      math_ops.reduce_sum(
          math_ops.maximum(
              math_ops.multiply(loss_mat1, mask_positives), 0.0)),
      num_positives, name='triplet_semihard_loss1')

    triplet_loss2 = math_ops.truediv(
      math_ops.reduce_sum(
          math_ops.maximum(
              math_ops.multiply(loss_mat2, mask_positives), 0.0)),
      num_positives, name='triplet_semihard_loss2')

    return triplet_loss + 0.5 * (triplet_loss1 + triplet_loss2)

if __name__ == '__main__':
    # test
    labels = np.array([[0, 1, 0, 1],
                       [0, 0, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 1, 1],
                       [1, 0, 0, 0]
                      ])
    embeddings1 = np.array([[0.20251631, 0.49964871, 0.31357543, 0.99332346, 0.40536699,
                            0.05654062, 0.07307319, 0.2950833, 0.5154805, 0.43801481],
                           [0.05170506, 0.9920793, 0.50820659, 0.80957615, 0.59039356,
                            0.83899964, 0.3024558, 0.29522561, 0.90828209, 0.7059259],
                           [0.06045745, 0.73130719, 0.81188, 0.37673241, 0.41282683,
                            0.00261911, 0.54569239, 0.52696678, 0.94666249, 0.4798159],
                           [0.9031102, 0.09828223, 0.67050717, 0.77313736, 0.47979198,
                            0.93205683, 0.30714715, 0.6665816, 0.11693463, 0.75662641],
                           [0.13010331, 0.70302084, 0.29719897, 0.4037086, 0.60219295,
                            0.18917132, 0.0928293, 0.70829784, 0.6350869, 0.74187586]], dtype=np.float32)

    embeddings2 = np.array([[0.0516531, -0.49964871, 0.31357543, -0.99332346, 0.40536699,
                             0.05654062, 0.07307319, 0.2950833, 0.5154805, 0.43801481],
                            [0.05170506, -0.92920793, 0.50820659, 0.80957615, 0.5039356,
                             0.83899964, 0.304558, 0.29522561, -0.90828209, 0.7059259],
                            [0.06045745, 0.73130719, 0.81192888, 0.37673241, 0.41282683,
                             0.00261911, -0.54569239, 0.5269668, 0.9466649, 0.478159],
                            [-0.9031102, 0.09828223, -0.67050717, -0.77313736, 0.4979198,
                             0.93205683, 0.30714715, 0.66625816, 0.11693463, 0.75662641],
                            [-0.13010331, 0.70302084, 0.2971997, 0.4037086, 0.60219295,
                             0.18917132, 0.0928293, 0.70829784, 0.6350869, 0.74187586]], dtype=np.float32)
    margin = 0.4
    label = labels[:,:]
    s = pairwise_distances(np.array(label,dtype=np.float32),np.array(label,dtype=np.float32))
    b = triplet_semihard_loss(embeddings1, embeddings2, label,label, margin)
    with tf.Session() as sess:
        print(sess.run(b))