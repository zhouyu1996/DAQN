#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/7/8 11:05
# @Author  : zhouyu
# @content : 
# @File    : dataset.py
# @Software: PyCharm

import numpy as np
from config import *

# this part codes reference https://github.com/caoyue10/aaai17-cdq
class Dataset(object):
    def __init__(self, data, label, code_dim, output_dim):
        print('===Initalizing Dataset===')
        self.dataset = data
        self.labels = label
        self.n_samples = self.labels.shape[0]
        self.codes = np.zeros((self.n_samples, code_dim))
        self.output = np.zeros((self.n_samples, output_dim), dtype=np.float32)

        self._perm = np.arange(self.n_samples)
        self._index_in_epoch = 0
        self._epochs_complete = 0
        print('===Dataset already===')
        return

    def get_labels(self):
        return self.labels

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self.n_samples:
            self._epochs_complete += 1
            # Shuffle the data
            np.random.shuffle(self._perm)
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch

        data = self.dataset[self._perm[start: end]]
        label = self.labels[self._perm[start: end]]
        return (data, label,
                self.codes[self._perm[start: end], :])

    def next_batch_data(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self.n_samples:
            self._epochs_complete += 1
            # Shuffle the data
            np.random.shuffle(self._perm)
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch

        data = self.dataset[self._perm[start: end]]
        label = self.get_labels[self._perm[start: end]]
        return (data, label)

    def finish_epoch(self):
        start = 0
        np.random.shuffle(self._perm)

    def next_batch_output_codes(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self.n_samples:
            # Shuffle the data
            np.random.shuffle(self._perm)
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch

        return (
            self.output[self._perm[start: end], :],
            self.codes[self._perm[start: end], :],
        )

    def full_dataset_without_shuffle(self):
        return self.dataset

    def full_output_without_shuffle(self):
        return self.output

    def full_codes_without_shuffle(self):
        return self.codes

    def copy_codes(self, codes):
        self.codes = codes
        return

    def copy_batch_codes(self, codes, batch_size):
        start = self._index_in_epoch - batch_size
        end = self._index_in_epoch
        self.codes[self._perm[start: end], :] = codes
        return

    def copy_batch_output(self, output1, batch_size):
        start = self._index_in_epoch - batch_size
        end = self._index_in_epoch
        self.output[self._perm[start: end], :] = output1
        return

    def n(self):
        return self.n_samples
