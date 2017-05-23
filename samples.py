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

def load_pavias(path):
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



