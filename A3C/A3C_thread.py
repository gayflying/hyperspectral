# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random
import time
import sys

from A3C_network import GameACFFNetwork
from A3C_config import *
import utils
import gym

class A3CTrainingThread(object):
    def __init__(self,
                 thread_index,
                 global_network,
                 initial_learning_rate,
                 learning_rate_input,
                 optimizer,
                 max_global_time_step,
                 device):

        self.thread_index = thread_index
        self.learning_rate_input = learning_rate_input
        self.max_global_time_step = max_global_time_step
        self.terminal_end = True

        self.local_network = GameACFFNetwork(args.action_size, thread_index, device)

        self.local_network.prepare_loss(args.entropy_beta)

        with tf.device(device):
            self.opt = optimizer
            local_gradients = self.opt.compute_gradients(self.local_network.total_loss, self.local_network.get_vars())
            self.gradients = [(tf.clip_by_norm(local_gradients[i][0], args.grad_norm_clip), global_network.get_vars()[i]) for i in range(len(local_gradients))]

        # update the global network using local gradients
        self.apply_gradients = self.opt.apply_gradients(self.gradients)

        self.sync = self.local_network.sync_from(global_network)

        self.env = utils.smart_env()

        self.local_t = 0

        self.initial_learning_rate = initial_learning_rate

        self.episode_reward = 0

        # variable controling log output
        self.prev_local_t = 0

    def _anneal_learning_rate(self, global_time_step):
        '''
        the learning rate anneal globally
        '''
        learning_rate = self.initial_learning_rate * (self.max_global_time_step - global_time_step) / self.max_global_time_step
        if learning_rate < 0.0:
            learning_rate = 0.0
        return learning_rate

    def choose_action(self, pi_values):
        # return the index of action
        return np.random.choice(range(len(pi_values)), p=pi_values)

    def _record_score(self, sess, summary_writer, summary_op, score_input, score, global_t):
        summary_str = sess.run(summary_op, feed_dict={
            score_input: score
        })
        summary_writer.add_summary(summary_str, global_t)
        summary_writer.flush()

    def set_start_time(self, start_time):
        self.start_time = start_time

    def process(self, sess, global_t, summary_writer, summary_op, score_input):
        states = []
        actions = []
        rewards = []
        values = []

        # copy weights from shared to local
        sess.run( self.sync )

        start_local_t = self.local_t

        if self.terminal_end:
            self.state = self.env.reset()
            self.terminal_end = False

        # t_max times loop
        for i in range(args.local_t_max):
            pi_, value_ = self.local_network.run_policy_and_value(sess, self.state)
            action_index = self.choose_action(pi_)
            action = args.action_map[action_index]

            states.append(self.state)
            actions.append(action_index)
            values.append(value_)

            if (self.thread_index == 0) and (self.local_t % args.log_interval == 0):
                print("pi={}".format(pi_))
                print(" V={}".format(value_))

            # process game and receive game result
            # state contain the next-step-state
            self.state, reward, self.terminal_end = self.env.next(action)

            self.episode_reward += reward

            # clip reward
            rewards.append( np.clip(reward, -1, 1) )

            self.local_t += 1

            if self.terminal_end:
                print("score={}".format(self.episode_reward))
                self._record_score(sess, summary_writer, summary_op, score_input,
                                    self.episode_reward, global_t)
                self.episode_reward = 0
                break

        if self.terminal_end:
            R = 0.0
        else:
            R = self.local_network.run_value(sess, self.state)

        actions.reverse()
        states.reverse()
        rewards.reverse()
        values.reverse()

        batch_si = []
        batch_a = []
        batch_td = []
        batch_R = []

        # compute and accmulate gradients
        for(ai, ri, si, Vi) in zip(actions, rewards, states, values):
            R = ri + args.gamma * R
            td = R - Vi
            a = np.zeros([args.action_size])
            a[ai] = 1

            batch_si.append(si)
            batch_a.append(a)
            batch_td.append(td)
            batch_R.append(R)

        cur_learning_rate = self._anneal_learning_rate(global_t)

        sess.run( self.apply_gradients,
                feed_dict = {   self.local_network.s: batch_si,
                                self.local_network.a: batch_a,
                                self.local_network.td: batch_td,
                                self.local_network.r: batch_R,
                                self.learning_rate_input: cur_learning_rate} )

        if (self.thread_index == 0) and (self.local_t - self.prev_local_t >= args.performance_log_interval):
            self.prev_local_t += args.performance_log_interval
            elapsed_time = time.time() - self.start_time
            steps_per_sec = global_t / elapsed_time
            print("### Performance : {} STEPS in {:.0f} sec. {:.0f} STEPS/sec. {:.2f}M STEPS/hour".format(
                global_t, elapsed_time, steps_per_sec, steps_per_sec * 3600 / 1000000.))

        # return how many steps processed in this loop
        diff_local_t = self.local_t - start_local_t
        return diff_local_t

