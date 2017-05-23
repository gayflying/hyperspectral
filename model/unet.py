#

import sys
sys.path.append('..')

import samples as tool_sample
import numpy as np
import pandas as pd
from collections import Counter

g_classnum = 9
g_size = 32



def generate_origin_data_pavias(path):
    data_origin_x,data_origin_y = tool_sample.load_pavias(path)
    samples_dict = tool_sample.get_samples_pavias(data_origin_x,data_origin_y)
    train_dict,test_dict = tool_sample.random_split_train_test(samples_dict,tool_sample.pavias_train_dict)
    rows,cols,channels = data_origin_x.shape
    data_y = np.zeros((rows,cols,g_classnum),dtype=np.uint16)
    for classsign,val_list in train_dict.items():
        row_indices_list = []
        col_indices_list = []
        for temp_row,temp_col in val_list:
            row_indices_list.append(temp_row)
            col_indices_list.append(temp_col)
        data_y[row_indices_list,col_indices_list,classsign-1] = classsign

    print('generate origin data of pavias done! x shape is ',data_origin_x.shape,' y shape is ', data_y.shape)
    return data_origin_x,data_y


def preprocessing(data_x):
    

def get_patches(data_x, data_y, amt=10000, aug=True):
    '''
    data agumentation
    :param data_x:
    :param data_y:
    :param amt:
    :param aug:
    :return:
    '''



# generate_origin_data_pavias('../data/hyper_data')
