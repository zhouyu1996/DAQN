#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/6/8 11:50
# @Author  : yu zhou
# @Site    : 
# @File    : data_handler.py
# @content :
# @Software: PyCharm

import h5py
import numpy as np
import cv2

TRAINING_SIZE = 4000
QUERY_SIZE = 1000
DATABASE_SIZE = 19015

# FLICKR urls
mir_path = '/data/zydata/flickr25k/FLICKR-25K.mat'

class DataSet(object):
    def __init__(self):
        self.load = False
        file = h5py.File(mir_path, 'r')
        orign_image = file['images'][:].transpose(0, 3, 2, 1).astype(np.uint8)
        # 20015  images  224*224*3(rgb)
        # 20015  tags    1386d
        # 20015  labels  24d
        images = np.zeros((orign_image.shape[0], 256, 256, 3))
        # resize images
        for i in range(orign_image.shape[0]):
            # 224*224->256*256 rgb
            itmp = cv2.resize(orign_image[i], (256, 256))
            images[i] = itmp
        del orign_image
        labels = file['LAll'][:]
        tags = file['YAll'][:]
        file.close()
        print(images.shape, tags.shape, labels.shape)
        self.traindata = {}
        self.testdata = {}
        self.dbdata = {}
        self.label = labels
        self.testdata[1] = images[0:QUERY_SIZE, :, :, :]
        self.testdata[2] = tags[0:QUERY_SIZE, :]
        self.testdata[3] = labels[0:QUERY_SIZE, :]

        self.traindata[1] = images[QUERY_SIZE:TRAINING_SIZE + QUERY_SIZE, :, :, :]
        self.traindata[2] = tags  [QUERY_SIZE:TRAINING_SIZE + QUERY_SIZE, :]
        self.traindata[3] = labels[QUERY_SIZE:TRAINING_SIZE + QUERY_SIZE, :]

        self.dbdata[1] = images[QUERY_SIZE:DATABASE_SIZE + QUERY_SIZE, :, :, :]
        self.dbdata[2] = tags  [QUERY_SIZE:DATABASE_SIZE + QUERY_SIZE, :]
        self.dbdata[3] = labels[QUERY_SIZE:DATABASE_SIZE + QUERY_SIZE, :]
        if (images.shape[0] == QUERY_SIZE + DATABASE_SIZE):
            self.load = True
            del images, tags, labels
            print("dataset split successfully!")
        else:
            self.load = True
            print('dataset split error!')

    # return a dic including the training data
    def import_train(self):
        if (self.load == False):
            return None
        return self.traindata

    # return a dic including the testing data
    def test_data(self):
        if (self.load == False):
            return None
        return self.testdata

    # return a dic including the db data
    def db_data(self):
        if (self.load == False):
            return None
        return self.dbdata


if __name__ == '__main__':
    # test code
    a  = DataSet()
    t1 = a.import_train()
    ind = 2313
    img = t1[1]
    tmp = img[ind]
    print(img.shape)
    # print(img[1])
    txt = t1[2]
    label = t1[3]
    print(np.array(txt[ind]).min(), np.array(txt[ind]).max())
    print(label[ind])
    cv2.imshow("org", img[ind])
    cv2.imshow("change", tmp[:,:,::-1])
    cv2.waitKey(0)
