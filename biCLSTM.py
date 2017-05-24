#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 20 16:09:06 2017
╋╋╋╋╋╋╋┏┓╋╋╋╋┏┓ @coypright: radar_bear
╋╋╋╋╋╋╋┃┃╋╋╋╋┃┃ leimingda@pku.edu.cn
┏━┳━━┳━┛┣━━┳━┫┗━┳━━┳━━┳━┓
┃┏┫┏┓┃┏┓┃┏┓┃┏┫┏┓┃┃━┫┏┓┃┏┛
┃┃┃┏┓┃┗┛┃┏┓┃┃┃┗┛┃┃━┫┏┓┃┃
┗┛┗┛┗┻━━┻┛┗┻┛┗━━┻━━┻┛┗┻┛
@author: leimingda
"""
# Thanks for the blog http://blog.csdn.net/jerr__y/article/details/61195257
import tensorflow as tf
import numpy as np
import datetime as dt
import time
from tensorflow.contrib import rnn
from tensorflow.python.ops import  array_ops
import model.CLSTMCell as CLSTM
import samples

'''
Parameters note

the output of CLSTM is a image with
SIZE "patch_height * patch_width"
CHANNEL "output_channels"

the input of CLSTM is a image patch with
SIZE "patch_height * patch_width"
CHANNEL "num_steps"

lr -> learning rate
log_steps -> steps per log
drop_out -> how many nodes will be kept
'''
output_channels = 4
multi = False
layer_num = 2
class_num = 9
num_steps = 103
patch_size = 16
patch_height = patch_size
patch_width = patch_size
batch_size_train = 32
batch_size_test = 2000
lr = 1e-3
drop_out = 0.5
log_steps = 100
'''
TODO
1. complete multi layer conv
2. multi thread data reading
'''
def inference(images, labels, batch_size, keep_prob):
    with tf.name_scope('pCLSTM'):
        p_clstm_cell = CLSTM.CLSTMCell(num_units=output_channels, state_is_tuple=True)
        init_state = p_clstm_cell.zero_state(batch_size, patch_height, patch_width)
        p_clstm_cell = rnn.DropoutWrapper(p_clstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
        if multi:
            p_clstm_cell = rnn.MultiRNNCell([p_clstm_cell]*layer_num, state_is_tuple=True)

        # outputs has shape [batch, time_step, height, width, output_channels]
        # the dynamic_rnn can only use one-dimension data, we have to write our own calculation loop
        # outputs, state = tf.nn.dynamic_rnn(clstm_cell, inputs=images, initial_state=init_state, time_major=False)
        state = init_state
        p_h_state = None
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (p_cell_output, state) = p_clstm_cell(images[:, time_step, :, :, np.newaxis], state)
                if time_step == num_steps-1:
                    p_h_state = p_cell_output
        # flat h_state
        p_h_flat = tf.reshape(p_h_state, [batch_size, -1])

    with tf.name_scope('nCLSTM'):
        n_clstm_cell = CLSTM.CLSTMCell(num_units=output_channels, state_is_tuple=True)
        init_state = n_clstm_cell.zero_state(batch_size, patch_height, patch_width)
        n_clstm_cell = rnn.DropoutWrapper(n_clstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
        if multi:
            n_clstm_cell = rnn.MultiRNNCell([n_clstm_cell]*layer_num, state_is_tuple=True)

        # outputs has shape [batch, time_step, height, width, output_channels]
        # the dynamic_rnn can only use one-dimension data, we have to write our own calculation loop
        # outputs, state = tf.nn.dynamic_rnn(clstm_cell, inputs=images, initial_state=init_state, time_major=False)
        state = init_state
        n_h_state = None
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (n_cell_output, state) = n_clstm_cell(images[:, num_steps-time_step-1, :, :, np.newaxis], state)
                if time_step == num_steps-1:
                    n_h_state = n_cell_output
        # flat h_state
        n_h_flat = tf.reshape(n_h_state, [batch_size, -1])

    with tf.name_scope('CF'):
        hidden_num = patch_size*patch_size*output_channels*2
        h_flat = array_ops.concat([p_h_flat,n_h_flat], axis=1)
        w = tf.Variable(tf.truncated_normal([hidden_num, class_num], stddev=0.1), dtype=tf.float32)
        b = tf.Variable(tf.constant(0.1,shape=[class_num]), dtype=tf.float32)
        y = tf.nn.softmax(tf.matmul(h_flat, w)+b)

    with tf.name_scope('Eval'):
        cross_entropy = -tf.reduce_mean(labels*tf.log(y))
        correct = tf.equal(tf.argmax(y,1), tf.argmax(labels,1))
        accuracy = tf.reduce_mean(tf.cast(correct, "float"))

    train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
    return train_op, accuracy, cross_entropy

def run_training():
    with tf.Graph().as_default():
        images = tf.placeholder("float", shape=[None, None, patch_height, patch_width])
        labels = tf.placeholder("float", shape=[None, class_num])
        paviaU = samples.paviaU_dataset()
        batch_size = tf.placeholder(tf.int32)
        keep_prob = tf.placeholder(tf.float32)
        train_op, accuracy, loss_op = inference(images, labels, batch_size, keep_prob)
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        start_time = dt.datetime.now()
        with tf.Session() as sess:
            sess.run(init_op)
            step = 1
            tic = time.time()
            while step<5000:
                batch = paviaU.next_batch(batch_size_train, train=True, patch_size=patch_size)
                sess.run(train_op, feed_dict={images: batch[0], labels: batch[1], batch_size: batch_size_train, keep_prob: drop_out})
                if step%log_steps == 0:
                    toc = time.time()
                    duration = toc-tic
                    speed = batch_size_train*log_steps/duration

                    batch = paviaU.next_batch(batch_size_test, train=False, patch_size=patch_size)
                    loss_value, accuracy_value = sess.run([loss_op, accuracy], feed_dict={images: batch[0], labels: batch[1], batch_size: batch_size_test, keep_prob:1.0})

                    print("%s step %d speed: %.3f frames/sec \n loss is %.3f test accuracy is:%.3f" % (dt.datetime.now()-start_time, step, speed, loss_value, accuracy_value))

                    tic = time.time()
                step += 1

if __name__ == '__main__':
    run_training()