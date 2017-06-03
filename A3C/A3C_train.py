# -*- coding: utf-8 -*-
import tensorflow as tf
import threading
import numpy as np
import random
import math
import os
import time
import signal

from A3C_network import GameACFFNetwork, GameACLSTMNetwork
from A3C_thread import A3CTrainingThread
from A3C_config import *


import utils
import gym


def log_uniform(lo, hi, rate):
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    v = log_lo * (1-rate) + log_hi * rate
    return math.exp(v)

device = "/cpu:0"
if args.use_gpu:
    device = "/gpu:0"

initial_learning_rate = log_uniform(args.initial_alpha_low, args.initial_alpha_high, args.initial_alpha_log_rate)

global_t = 0

stop_requested = False

if args.use_lstm:
    global_network = GameACLSTMNetwork(args.action_size, -1, device)
else:
    global_network = GameACFFNetwork(args.action_size, -1, device)

learning_rate_input = tf.placeholder("float")

grad_applier = tf.train.RMSPropOptimizer(learning_rate = learning_rate_input,
                                decay = args.rmsp_alpha,
                                momentum = 0.0,
                                epsilon = args.rmsp_epsilon)

training_threads = []
for i in range(args.thread_num):
    thread = A3CTrainingThread(i, global_network, initial_learning_rate, learning_rate_input, grad_applier, args.max_time_step, device=device)
    training_threads.append(thread)

# prepare session
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
with tf.Session() as sess:
    sess.run(init_op)

    # summary for tensorboard
    score_input = tf.placeholder(tf.int32)
    tf.summary.scalar("score", score_input)

    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(args.log_file, sess.graph)

    # init or load checkpoint with saver
    saver = tf.train.Saver()

    if args.use_chechpoint:
        checkpoint = tf.train.get_checkpoint_state(args.checkpoint_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("checkpoint loaded:", checkpoint.model_checkpoint_path)
            tokens = checkpoint.model_checkpoint_path.split("-")
            # set global step
            global_t = int(tokens[1])
            print(">>> global step set: ", global_t)
            # set wall time
            wall_t_fname = args.checkpoint_dir + '/' + 'wall_t.' + str(global_t)
            with open(wall_t_fname, 'r') as f:
                wall_t = float(f.read())
        else:
            print("Could not find old checkpoint")
            # set wall time
            wall_t = 0.0
    else:
        wall_t = 0.0


    def train(thread_index):
        global global_t
        training_thread = training_threads[thread_index]
        # set start_time
        start_time = time.time() - wall_t
        training_thread.set_start_time(start_time)

        while True:
            if stop_requested:
                break
            if global_t > args.max_time_step:
                break
            diff_global_t = training_thread.process(sess, global_t, summary_writer, summary_op, score_input)
            global_t += diff_global_t

    def signal_handler(signal, frame):
        global stop_requested
        print('You pressed Ctrl+C!')
        stop_requested = True

    train_threads = []
    for i in range(args.thread_num):
        train_threads.append(threading.Thread(target=train, args=(i,)))

    # set start time
    start_time = time.time() - wall_t

    for t in train_threads:
        t.start()

    print('Press Ctrl+C to stop')
    signal.signal(signal.SIGINT, signal_handler)
    signal.pause()

    print('Now saving data. Please wait')

    for t in train_threads:
        t.join()

    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)

    # write wall time
    wall_t = time.time() - start_time
    wall_t_fname = args.checkpoint_dir + '/' + 'wall_t.' + str(global_t)
    with open(wall_t_fname, 'w') as f:
        f.write(str(wall_t))
    # write checkpoint
    saver.save(sess, args.checkpoint_dir + '/' + 'checkpoint', global_step = global_t)

