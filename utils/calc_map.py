#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/6/8 11:13
# @Author  : yu zhou
# @Site    : 
# @File    : calc_map.py
# @content :
# @Software: PyCharm

import numpy as np

class MAPs:
    def __init__(self, C, subspace_num, subcenter_num, R):
        self.C = C
        self.subspace_num = subspace_num
        self.subcenter_num = subcenter_num
        self.R = R
        self.dis = []
        for i in range(subcenter_num * subspace_num):
            self.dis.append([])
            for j in range(subcenter_num * subspace_num):
                self.dis[i].append(self.vdistance(C[i, :], C[j, :]))
        self.dis = np.array(self.dis)

    def vdistance(self, a, b):
        return np.dot(a, b.T)

    def  mdistance(self, a, b):
        return np.dot(a, b.T)

    # Use Asymmetric Quantizer Distance (AQD) to computer map@R for evaluation
    def get_mAPs_AQD(self, database, query):
        self.all_rel = self.mdistance(query.output, np.dot(database.codes, self.C))
        ids = np.argsort(-self.all_rel, 1)
        APx = []
        query_labels = query.get_labels()
        database_labels = database.get_labels()
        for i in range(self.all_rel.shape[0]):
            label = query_labels[i, :]
            label[label == 0] = -1
            idx = ids[i, :]
            imatch = np.sum(database_labels[idx[0: self.R], :] == label, 1) > 0
            rel = np.sum(imatch)
            Lx = np.cumsum(imatch)
            Px = Lx.astype(float) / np.arange(1, self.R + 1, 1)
            if rel != 0:
                APx.append(np.sum(Px * imatch) / rel)
            if i % 100 == 0:
                print("step: ", i)
        print("mAPs: ", np.mean(np.array(APx)))
        return np.mean(np.array(APx))




