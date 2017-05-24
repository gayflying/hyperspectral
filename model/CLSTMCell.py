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
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.ops import  array_ops
from tensorflow.python.ops.math_ops import sigmoid, tanh
from tensorflow.python.ops import variable_scope as vs

def _default_conv(args, output_size, k_size=3,
          bias=True, bias_start=0.0, scope=None):
    '''
    args should be a list, elements in args should have shape
    [batch, height, width, channel]
    all elements should have same batch, height and width
    the num of channel may different
    '''
    # check if the args shapes are satisfied
    shapes = [a.get_shape().as_list() for a in args]
    batch_size = shapes[0][0]
    height = shapes[0][1]
    width  = shapes[0][2]
    channels = 0
    for shape in shapes:
        if len(shape) != 4:
            raise ValueError("Conv is expecting [batch, height, width, channels], but args shape: %s" % str(shapes))
        if not shape[3]:
            raise ValueError("Conv expects shape[3] of arguments: %s" % str(shapes))
        if shape[0] != batch_size:
            raise ValueError("Inconsistent batch size in arguments: %s" % str(shapes))
        if shape[1] == height and shape[2] == width:
            channels += shape[3]
        else :
            raise ValueError("Inconsistent height and width size in arguments: %s" % str(shapes))

    # a single conv layer
    with vs.variable_scope(scope or "Conv"):
        kernel = vs.get_variable("Kernel", [k_size, k_size, channels, output_size])
    if len(args) == 1:
      res = tf.nn.conv2d(args[0], kernel, [1, 1, 1, 1], padding='SAME')
    else:
      res = tf.nn.conv2d(array_ops.concat(args, axis=3), kernel, [1, 1, 1, 1], padding='SAME')
    # add bias onto the conv result
    if not bias:
        return res
    else:
        bias_term = vs.get_variable("Bias", [output_size], initializer=tf.constant_initializer(bias_start))
        return res + bias_term

class  CLSTMCell(rnn.RNNCell):
    """
    Inputs [batch, height, width, channels]
    Outputs [batch, height, width, num_units]
    The implementation is based on http://arxiv.org/abs/1506.04214.
    The key is two function:
    1. __call__
        one should reload __call__ in the standard way to make it work well with rnn.dynamic_rnn and other builtin functions
        Note: pay attention to param state_is_tuple, the builtin function use tuple but not tensor when delivery state across steps
    2. zero_state
        this function should return an all-zero tensor in the shape of state. if state_is_tuple, it should return a rnn.LSTMStateTuple
    TODO
    let _conv can be modified outside the class
    let users can use their own conv network
    """
    def __init__(self, num_units, input_size=None,
               forget_bias=1.0, state_is_tuple=False,
               activation=tanh, conv=_default_conv):
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation
        self._conv = conv

    @property
    def state_size(self):
        return (rnn.LSTMStateTuple(self._num_units, self._num_units)
                if self._state_is_tuple else 2 * self._num_units)
    @property
    def output_size(self):
        return self._num_units

    def zero_state(self, batch_size, height, width):
        if self._state_is_tuple:
            return rnn.LSTMStateTuple(
                    tf.zeros([batch_size, height, width, self._num_units]),
                    tf.zeros([batch_size, height, width, self._num_units])
                    )
        else:
            return tf.zeros([batch_size, height, width, self._num_units*2])

    def __call__(self, inputs, state, k_size=3, scope=None):
        '''
        inputs has shape [batch, height, width, channels]
        if state is a tuple, it should be (c,h)
        where c and h both have shape [batch, height, width, output_size]
        if state is a tenser, shape [batch, height, width, 2*output_size]
        the Conv kernel has size k_size*k_size
        '''
        with vs.variable_scope(scope or type(self).__name__):
            # if state is a tuple, it is (state-c, state-h)
            # if it is a tensor, its shape is (batch, height, width, channel)
            # the first half channels are state-c, while the others are state-h
            if self._state_is_tuple:
                c, h = state
            else:
                c, h = array_ops.split(state, num_or_size_splits=2, axis=3)

            # in LSTM, the input has four copy
            # 1. for forget gate -> f
            # 2. for input gate -> i
            # 3. for update state-c value -> j
            # 4. for output -> o
            # each copy has its own conv network, we generate them together, that's why the concat has channel num 4*_num_units
            # then we split concat along the channel axis
            # inputs has shape [batch, height, width, channels]
            # h has shape [batch, height, width, output_size]
            concat = self._conv([inputs, h], output_size=4*self._num_units, bias=True)
            f, i, j, o = array_ops.split(concat, num_or_size_splits=4, axis=3)

            # more infomation about state update see
            # http://www.jianshu.com/p/9dc9f41f0b29
            new_c = c * sigmoid(f + self._forget_bias) + self._activation(j) * sigmoid(i)
            new_h = self._activation(new_c) * sigmoid(o)

            if self._state_is_tuple:
                new_state = rnn.LSTMStateTuple(new_c, new_h)
            else:
                new_state = array_ops.concat([new_c, new_h], 3)

            return new_h, new_state






