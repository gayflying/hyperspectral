#

import numpy as np
import pandas as pd
import scipy.io as sio
from collections import defaultdict


pavias_train_dict = {
    1:548,
    2:540,
    3:392,
    4:524,
    5:265,
    6:532,
    7:375,
    8:514,
    9:231,
}

def load_pavias(path='Pavia', std=True):
    if std:
        path_data_x = path + '/PaviaU_std.mat'
    else:
        path_data_x = path + '/PaviaU.mat'
    path_data_y = path + '/PaviaU_gt.mat'
    data_x = sio.loadmat(path_data_x)
    data_y = sio.loadmat(path_data_y)
    data_x = data_x['paviaU']
    data_y = data_y['paviaU_gt']
    print('load pavias done! ')
    print('x shape is ',data_x.shape)
    print('y shape is ', data_y.shape)
    return data_x,data_y

def get_samples_pavias(data_x,data_y):
    '''
    :param path:
    :return:dict which key is the classsign and value is list contains tupels(row,col)
    {1:[(23,23),(23,24),..],2:[(25,26)]}
    '''
    samples = defaultdict(list)
    rows,cols = data_y.shape
    for i in range(rows):
        for j in range(cols):
            if data_y[i,j] > 0:
                samples[data_y[i,j]].append((i,j))

    for key,val in samples.items():
        print('%d:\t%d' % (key,len(val)))

    print('trans pavia done!')
    return samples

def random_split_train_test(samples, per_train_size_dict):
    '''
    random split the given samples and split it into two parts which contains
    trainset and testset
    trainset : testset = 1 : 9 in each class
    :param samples: samples
    :return: (dict_train,dict_test)
    '''
    train_dict = defaultdict(list)
    test_dict = defaultdict(list)
    for key,value in samples.items():
        csize = len(value)
        train_size = per_train_size_dict[key]
        selected_index = np.random.choice(list(range(csize)),train_size,False)
        for index,item in enumerate(value):
            if index in selected_index:
                train_dict[key].append(item)
            else:
                test_dict[key].append(item)
    for key in train_dict.keys():
        print('class=%d, sum=%d, train=%d, test=%d' %(key,len(samples[key]),len(train_dict[key]),len(test_dict[key])))
    return train_dict,test_dict

def get_patch_cube(data_x, position, patch_size=16):
    '''
    Padding mode: SAME
    '''
    height, width = position
    max_h, max_w, max_c = data_x.shape
    h_start = height-int(patch_size/2)
    h_end = h_start+patch_size
    w_start = width-int(patch_size/2)
    w_end = w_start+patch_size
    if h_start>=0 and w_start>=0 and h_end<=max_h and w_end<=max_w:
        cube = data_x[h_start:h_end, w_start:w_end, :]
    else:
        cube = np.zeros([patch_size, patch_size, max_c])
        for i in range(patch_size):
            for j in range(patch_size):
                if h_start+i>=0 and h_start+i<max_h and w_start+j>=0 and w_start+j<max_w:
                    cube[i,j,:] = data_x[h_start+i, w_start+j, :]
    return cube

def get_patch_cubes(data_x, position_list, patch_size=16):
    cubes = []
    for position in position_list:
        cubes.append(get_patch_cube(data_x, position, patch_size=patch_size))
    return cubes

class paviaU_dataset():
    def __init__(self, path='Pavia', std=True):
        self._x, self._y = load_pavias(path=path, std=std)
        # self._sample_position is a dict which key is the classsign and value is list contains tupels(row,col)
        # {1:[(23,23),(23,24),..],2:[(25,26)]}
        self._sample_position = get_samples_pavias(self._x, self._y)
        self.train_dict, self.test_dict = random_split_train_test(self._sample_position, pavias_train_dict)
        self.dict_keys = list(self.train_dict.keys())

    def next_batch(self, batch_size, one_hot=True, train=True, patch_size=16):
        if train:
            dict = self.train_dict
        else:
            dict = self.test_dict
        key_batch = []
        cube_batch = []
        for _ in range(batch_size):
            key = np.random.choice(self.dict_keys)
            sample_index = np.random.choice(list(range(len(dict[key]))))
            cube = get_patch_cube(self._x, dict[key][sample_index], patch_size=patch_size)
            cube_batch.append(cube)
            if one_hot:
                one_hot_key = np.zeros(9)
                one_hot_key[key-1] = 1
                key = one_hot_key
            key_batch.append(key)
        # cube_batch have shape [batch, patch_size, patch_size, channel]
        # key_batch have shape [batch, classnum]
        # when output, transpose cube batch to [batch, channel, p, p]
        cube_batch = np.transpose(np.array(cube_batch), [0,3,1,2])
        key_batch = np.array(key_batch)
        return cube_batch, key_batch












