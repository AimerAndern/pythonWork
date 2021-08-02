import os
import numpy as np
import tensorflow as tf


class DeepQNetwork(object):
    def __init__(self, lr, n_actions, name, fcl_dims=256,
                 input_dims=(210, 160, 4), chkpt_dir='dqn'):
        self.lr = lr
        self.name = name
        self.n_actions = n_actions
        self.fcl_Dims = fcl_dims
        self.input_dims = input_dims
        self.sess = tf.Session()
        self.build_network()
        self.sess.run(tf.global_variables_initializer())
        self.saver=tf.train.Saver()
        self.checkpoint_file=os.path.join(chkpt_dir,'deepqnet.ckpt')
        self.params=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=self.name)



    def build_net(self):
        with tf.variable_scpoe(self.name)
            self.input = placeholder(tf.float32,shape = [None, *self.input_dims],
                                     name='inputs')
            self.actions=tf.placeholder(tf.float32,shape=[None,self.n_actions],
                                        name='action_taken')

            conv1 = tf.layers.conv2d(inputs=self.input, filters = 32,
                                     kernel_size=(8,8), strides = 4, name = 'conv1')
            conv1_activated = tf.nn.relu(conv1)
            conv2 = tf.layers.conv2d(inputs=conv1_activated, filters=64,
                                     kernel_size=(4, 4), strides=2, name='conv1')
            conv1 = tf.layers.conv2d(inputs=self.input, filters=32,
                                     kernel_size=(8, 8), strides=4, name='conv1')
