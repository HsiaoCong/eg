#coding:utf-8

import time
import numpy as np
import tensorflow as tf

from base import Model

try:
    linear=tf.nn.rnn_cell.linear
except:
  from tensorflow.python.ops.rnn_cell import _linear as linear


class NVDM(Model):
    """Neural Varational Document Model"""

    def __init__(self, sess, reader, dataset="ptb",
               decay_rate=0.96, decay_step=10000, embed_dim=500,
               h_dim=50, learning_rate=0.001, max_iter=450000,
               checkpoint_dir="checkpoint"):

        """
        Initialize Neural Varational Document Model.

        params:
        sess: TensorFlow Session object.
        reader: TextReader object for training and test.
        dataset: The name of dataset to use.
        h_dim: The dimension of document representations (h). [50, 200]
        """

        self.sess = sess
        self.reader = reader

        self.h_dim = h_dim
        self.embed_dim = embed_dim

        self.max_iter = max_iter
        self.decay_rate = decay_rate
        self.decay_step = decay_step
        self.checkpoint_dir = checkpoint_dir
        self.step = tf.Variable(0, trainable=False)
        self.lr = tf.train.exponential_decay(
            learning_rate, self.step, 10000, decay_rate, staircase=True, name="lr")

        _ = tf.summary.scalar("learning rate", self.lr)

        self.dataset = dataset
        self._attrs = ["h_dim", "embed_dim", "max_iter", "dataset",
                       "learning_rate", "decay_rate", "decay_step"]

        self.build_model()

    def build_model(self):
        self.x=tf.placeholder(tf.float32,[self.reader.vocab_size],name='input')
        self.x_idx=tf.placeholder(tf.int32,[None],name='x_idx')

        self.build_encoder()
        self.build_generator()

        # Kullback Leibler divergence
        self.e_loss = -0.5 * tf.reduce_sum(1 + self.log_sigma_sq - tf.square(self.mu) - tf.exp(self.log_sigma_sq))

        # Log likelihood
        self.g_loss = -tf.reduce_sum(tf.log(tf.gather(self.p_x_i, self.x_idx) + 1e-10))

        self.loss = self.e_loss + self.g_loss

        self.encoder_var_list, self.generator_var_list = [], []
        for var in tf.trainable_variables():
            if "encoder" in var.name:
                self.encoder_var_list.append(var)
            elif "generator" in var.name:
                self.generator_var_list.append(var)

        # optimizer for alternative update
        self.optim_e = tf.train.AdamOptimizer(learning_rate=self.lr) \
            .minimize(self.e_loss, global_step=self.step, var_list=self.encoder_var_list)
        self.optim_g = tf.train.AdamOptimizer(learning_rate=self.lr) \
            .minimize(self.g_loss, global_step=self.step, var_list=self.generator_var_list)

        # optimizer for one shot update
        self.optim = tf.train.AdamOptimizer(learning_rate=self.lr) \
            .minimize(self.loss, global_step=self.step)

        _ = tf.summary.scalar("encoder loss", self.e_loss)
        _ = tf.summary.scalar("generator loss", self.g_loss)
        _ = tf.summary.scalar("total loss", self.loss)






