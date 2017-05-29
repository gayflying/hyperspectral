# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from A3C_config import *


# Actor-Critic Network Base Class
# (Policy network and Value network)
class GameACNetwork(object):
    def __init__(self,
                             action_size,
                             thread_index, # -1 for global
                             device="/cpu:0"):
        self._action_size = action_size
        self._thread_index = thread_index
        self._device = device

    def prepare_loss(self, entropy_beta):
        with tf.device(self._device):
            # taken action (input for policy)
            self.a = tf.placeholder("float", [None, self._action_size])

            # temporary difference (R-V) (input for policy)
            self.td = tf.placeholder("float", [None])

            # avoid NaN with clipping when value in pi becomes zero
            log_pi = tf.log(tf.clip_by_value(self.pi, 1e-20, 1.0))

            # policy entropy
            entropy = -tf.reduce_sum(self.pi * log_pi, reduction_indices=1)

            # policy loss (output)  (Adding minus, because the original paper's objective function is for gradient ascent, but we use gradient descent optimizer.)
            policy_loss = - tf.reduce_sum( tf.reduce_sum( tf.multiply( log_pi, self.a ), reduction_indices=1 ) * self.td + entropy * args.entropy_beta )

            # R (input for value)
            self.r = tf.placeholder("float", [None])

            # value loss (output)
            # (Learning rate for Critic is half of Actor's, so multiply by 0.5)
            value_loss = 0.5 * tf.nn.l2_loss(self.r - self.v)

            # gradienet of policy and value are summed up
            self.total_loss = policy_loss + value_loss

    def run_policy_and_value(self, sess, s_t):
        raise NotImplementedError()

    def run_policy(self, sess, s_t):
        raise NotImplementedError()

    def run_value(self, sess, s_t):
        raise NotImplementedError()

    def get_vars(self):
        raise NotImplementedError()

    def sync_from(self, src_netowrk, name=None):
        '''
        return a list of ops
        run the list will sync self from src_network
        '''
        src_vars = src_netowrk.get_vars()
        dst_vars = self.get_vars()

        sync_ops = []

        with tf.device(self._device):
            with tf.name_scope(name, "GameACNetwork", []) as name:
                for(src_var, dst_var) in zip(src_vars, dst_vars):
                    sync_op = tf.assign(dst_var, src_var)
                    sync_ops.append(sync_op)

                return tf.group(*sync_ops, name=name)

    # weight initialization based on muupan's code
    # https://github.com/muupan/async-rl/blob/master/a3c_ale.py
    def _fc_variable(self, weight_shape):
        input_channels  = weight_shape[0]
        output_channels = weight_shape[1]
        d = 1.0 / np.sqrt(input_channels)
        bias_shape = [output_channels]
        weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
        bias   = tf.Variable(tf.random_uniform(bias_shape,   minval=-d, maxval=d))
        return weight, bias

    def _conv_variable(self, weight_shape):
        w = weight_shape[0]
        h = weight_shape[1]
        input_channels  = weight_shape[2]
        output_channels = weight_shape[3]
        d = 1.0 / np.sqrt(input_channels * w * h)
        bias_shape = [output_channels]
        weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
        bias   = tf.Variable(tf.random_uniform(bias_shape,   minval=-d, maxval=d))
        return weight, bias

    def _conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

# Actor-Critic FF Network
class GameACFFNetwork(GameACNetwork):
    def __init__(self,
                 action_size,
                 thread_index, # -1 for global
                 device="/cpu:0"):
        GameACNetwork.__init__(self, action_size, thread_index, device)

        scope_name = "net_" + str(self._thread_index)
        with tf.device(self._device), tf.variable_scope(scope_name) as scope:
            self.W_conv1, self.b_conv1 = self._conv_variable([8, 8, 4, 16])  # stride=4
            self.W_conv2, self.b_conv2 = self._conv_variable([4, 4, 16, 32]) # stride=2

            self.W_fc1, self.b_fc1 = self._fc_variable([3200, 512])

            # weight for policy output layer
            self.W_fc2, self.b_fc2 = self._fc_variable([512, action_size])

            # weight for value output layer
            self.W_fc3, self.b_fc3 = self._fc_variable([512, 1])

            # state (input)
            self.s = tf.placeholder("float", [None, 80, 80, 4])

            h_conv1 = tf.nn.relu(self._conv2d(self.s,  self.W_conv1, 4) + self.b_conv1)
            h_conv2 = tf.nn.relu(self._conv2d(h_conv1, self.W_conv2, 2) + self.b_conv2)

            h_conv2_flat = tf.reshape(h_conv2, [-1, 3200])
            h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, self.W_fc1) + self.b_fc1)

            # policy (output)
            self.pi = tf.nn.softmax(tf.matmul(h_fc1, self.W_fc2) + self.b_fc2)
            # value (output)
            v_ = tf.matmul(h_fc1, self.W_fc3) + self.b_fc3
            self.v = tf.reshape( v_, [-1] )

    def run_policy_and_value(self, sess, s_t):
        '''
        the s_t is a single state
        return the pi and value of this state
        '''
        pi_out, v_out = sess.run( [self.pi, self.v], feed_dict = {self.s : [s_t]} )
        return (pi_out[0], v_out[0])

    def run_policy(self, sess, s_t):
        '''
        the s_t is a single state
        return the pi of this state
        '''
        pi_out = sess.run( self.pi, feed_dict = {self.s : [s_t]} )
        return pi_out[0]

    def run_value(self, sess, s_t):
        '''
        the s_t is a single state
        return the value of this state
        '''
        v_out = sess.run( self.v, feed_dict = {self.s : [s_t]} )
        return v_out[0]

    def get_vars(self):
        '''
        return the list of all variables in this network
        '''
        return [self.W_conv1, self.b_conv1,
                        self.W_conv2, self.b_conv2,
                        self.W_fc1, self.b_fc1,
                        self.W_fc2, self.b_fc2,
                        self.W_fc3, self.b_fc3]