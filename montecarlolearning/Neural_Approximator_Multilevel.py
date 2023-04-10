#Packages
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from matplotlib import pyplot as plt 
import tensorflow_probability as tfp
import time
from tensorflow.python.ops import init_ops
from tensorflow.python.training.moving_averages import assign_moving_average

class Neural_Approximator_Multilevel:
        # Setter for data Generator
    def set_Generator(self, Generator):
        self._Generator = Generator
        
    @property
    def Generator(self):
        return self._Generator

    # Setter for hiddenNeurons
    def set_hiddenNeurons(self, hiddenNeurons):
        self._hiddenNeurons = hiddenNeurons

    # Setter for hiddenLayers
    def set_hiddenLayers(self, hiddenLayers):
        self._hiddenLayers = hiddenLayers
        
    @property
    def HiddenNeurons(self):
        return self._hiddenNeurons
    
    @property
    def HiddenLayers(self):
        return self._hiddenLayers

        
    @staticmethod
    def neural_net(x, xi_approx, neurons, is_training, name, net_id, mv_decay=0.9, dtype=tf.float32):
        def approx_test(): return xi_approx
        def approx_learn(): return x
        x = tf.cond(is_training, approx_learn, approx_test)

        def _batch_normalization(_x):
            beta = tf.get_variable(f'beta{net_id}', [_x.get_shape()[-1]], dtype, init_ops.zeros_initializer())
            gamma = tf.get_variable(f'gamma{net_id}', [_x.get_shape()[-1]], dtype, init_ops.ones_initializer())
            mv_mean = tf.get_variable(f'mv_mean{net_id}', [_x.get_shape()[-1]], dtype, init_ops.zeros_initializer(), trainable=False)
            mv_variance = tf.get_variable(f'mv_variance{net_id}', [_x.get_shape()[-1]], dtype, init_ops.ones_initializer(), trainable=False)
            mean, variance = tf.nn.moments(_x, [0], name=f'moments{net_id}')
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, assign_moving_average(mv_mean, mean, mv_decay, True))
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, assign_moving_average(mv_variance, variance, mv_decay, False))
            mean, variance = tf.cond(is_training, lambda: (mean, variance), lambda: (mv_mean, mv_variance))
            return tf.nn.batch_normalization(_x, mean, variance, beta, gamma, 1e-6)

        def _layer(_x, out_size, activation_fn):
            w = tf.get_variable(f'weights{net_id}', [_x.get_shape().as_list()[-1], out_size], dtype, tf.initializers.glorot_uniform())
            return activation_fn(_batch_normalization(tf.matmul(_x, w)))

        with tf.variable_scope(name):
            x = _batch_normalization(x)
            for i in range(len(neurons)):
                with tf.variable_scope(f'layer{net_id}_{i + 1}_'):
                    x = _layer(x, neurons[i], tf.nn.tanh if i < len(neurons)-1 else tf.identity)
        return x