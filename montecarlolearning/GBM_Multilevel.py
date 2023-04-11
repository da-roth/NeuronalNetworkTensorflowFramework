#Packages
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from matplotlib import pyplot as plt 
import tensorflow_probability as tfp
import time
from tensorflow.python.ops import init_ops
from tensorflow.python.training.moving_averages import assign_moving_average

class GBM_Multilevel:
    def __init__(self):
        self._stepsInitialLevel = 1
        
    def set_batch_sizes(self, value):
        self._batch_sizes = value

    def set_stepsPerLevel(self, value):
        self._stepsPerLevel = value

    def set_dtype(self, value):
        self._dtype = value

    def set_mc_samples_ref(self, value):
        self._mc_samples_ref = value

    def set_loop_var_mc(self, value):
        self._loop_var_mc = value

    @property
    def StepsInitialLevel(self):
        return self._stepsInitialLevel

    def set_stepsInitialLevel(self, value):
        self._stepsInitialLevel = value

    @staticmethod
    def phi(x,sigma,mu,T,K, axis=1):
        payoff=tf.exp(-mu * T)* tf.maximum(x - K, 0.)
        return payoff
    
    # First level: actually just P_0 without level estimator
    def Milstein_level0(self, idx, s,sigma,mu,T,K, samples): 
        if isinstance(self._stepsPerLevel, list):
            batchSizeFirstLevel = self._batch_sizes[0]
            stepsForFirstLevel = self._stepsPerLevel[0]
        else:
            batchSizeFirstLevel = self._batch_sizes
            stepsForFirstLevel = self._stepsPerLeve
        z = tf.random_normal(shape=(samples, batchSizeFirstLevel, 1),
                            stddev=1., dtype=self._dtype)
        h=T/stepsForFirstLevel
        s=s + mu *s * h +sigma * s *tf.sqrt(h)*z + 0.5 *sigma *s *sigma * ((tf.sqrt(h)*z)**2-h)
        return tf.add(idx, 1), s, sigma,mu,T,K
    
    def MonteCarlo_loop_level0(self, idx, p):
        if isinstance(self._stepsPerLevel, list):
            stepsForFirstLevel = self._stepsPerLevel[0]
        else:
            stepsForFirstLevel = self._stepsPerLevel
        _, _x, _sigma,_mu,_T,_K = tf.while_loop(lambda _idx, s, sigma,mu,T,K: _idx < stepsForFirstLevel,
                            lambda _idx, s, sigma,mu,T,K: self.Milstein_level0(_idx, s, sigma,mu,T,K,
                                                    self._mc_samples_ref),
                                                    self._loop_var_mc[0])
        return idx + 1, p + tf.reduce_mean(GBM_Multilevel.phi(_x,_sigma,_mu,_T,_K, 2), axis=0)  
        
    #Multilevel Monte Carlo level estimators
    def Milstein_levelEstimator(self, idx, s, sfine, sigma, mu, T, K, samples, level): 
        z1 = tf.random_normal(shape=(samples, self._batch_sizes[level], 1),
                            stddev=1., dtype=self._dtype)
        z2 = tf.random_normal(shape=(samples, self._batch_sizes[level], 1),
                            stddev=1., dtype=self._dtype)
        z=(z1+z2)/tf.sqrt(2.)
        amountSteps = self._stepsPerLevel[level-1]
        hcoarse= T / amountSteps
        hfine= T / (amountSteps * 2)
        sfine=sfine + mu *sfine * hfine +sigma * sfine *tf.sqrt(hfine)*z1 + 0.5 *sigma *sfine *sigma * ((tf.sqrt(hfine)*z1)**2-hfine)
        sfine=sfine + mu *sfine * hfine +sigma * sfine *tf.sqrt(hfine)*z2 + 0.5 *sigma *sfine *sigma * ((tf.sqrt(hfine)*z2)**2-hfine)
        s=s + mu *s * hcoarse +sigma * s *tf.sqrt(hcoarse)*z + 0.5 *sigma *s *sigma * ((tf.sqrt(hcoarse)*z)**2-hcoarse)    
        return tf.add(idx, 1), s, sfine, sigma, mu, T, K

    def MonteCarlo_loop_levelEstimator(self, idx, p, level):
        amountSteps = self._stepsPerLevel[level-1]
        _, _xcoarse, _xfine, sigma, mu, T, K = tf.while_loop(lambda _idx, s, xfine, sigma, mu, T, K: _idx < amountSteps,
                                                            lambda _idx, s, xfine, sigma, mu, T, K: self.Milstein_levelEstimator(_idx, s, xfine, sigma, mu, T, K, self._mc_samples_ref, level),
                                                            self._loop_var_mc[level])
        return idx + 1, p + tf.reduce_mean(GBM_Multilevel.phi(_xfine, sigma, mu, T, K, 2) - GBM_Multilevel.phi(_xcoarse, sigma, mu, T, K, 2), axis=0)