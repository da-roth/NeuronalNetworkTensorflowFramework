__version__ = '0.0.1'

def hello_world():
    print("This is my first pip package!")
    
    
    
#try:
#    %matplotlib notebook
#except Exception:
#    pass

# import and test
import tensorflow as tf2
#print("TF version =", tf2.__version__)

# we want TF 2.x
assert tf2.__version__ >= "2.0"

# disable eager execution etc
tf = tf2.compat.v1
tf.disable_eager_execution()

# disable annoying warnings
tf.logging.set_verbosity(tf.logging.ERROR)
import warnings
warnings.filterwarnings('ignore')

# make sure we have GPU support
#print("GPU support = ", tf.test.is_gpu_available())

# import other useful libs
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import time
from tqdm import tqdm_notebook

# representation of real numbers in TF, change here for 32/64 bits
real_type = tf.float32
# real_type = tf.float64

import sys
import os

# Data importation or generation classes
from montecarlolearning.BlackScholes import *
from montecarlolearning.DataImporter import *
from montecarlolearning.CDF import *
from montecarlolearning.GBM import *
from montecarlolearning.GBM_Differential import *
from montecarlolearning.GBM_5d import *
from montecarlolearning.Multilevel_GBM import *

# # Training and testing functions
#from montecarlolearning.train_and_test_with_differentials import *
from montecarlolearning.train_and_test import *
from montecarlolearning.TrainingOptionEnums import *

# # Plot function
from montecarlolearning.plot_results import *

# 
from montecarlolearning.Neural_Approximator import *
from montecarlolearning.vanilla_net import *
from montecarlolearning.backprop import *
from montecarlolearning.normalize_data import *
from montecarlolearning.vanilla_graph import *
from montecarlolearning.diff_training_graph import *
from montecarlolearning.train import *
from montecarlolearning.TrainingOptionEnums import *
from montecarlolearning.Neural_Approximator import *
from montecarlolearning.vanilla_net import *
from montecarlolearning.vanilla_net_biasNeuron import *
from montecarlolearning.backprop import *


# Multilevel
from montecarlolearning.train_and_test_Multilevel import *
from montecarlolearning.Neural_Approximator_Multilevel  import *
from montecarlolearning.GBM_Multilevel import *
