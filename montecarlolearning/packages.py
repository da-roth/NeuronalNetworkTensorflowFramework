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
print("GPU support = ", tf.test.is_gpu_available())

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
sys.path.append( os.path.join(mainDirectory,'montecarlolearning') )
#sys.path.append( os.path.join(mainDirectory,'montecarlolearning','DataImportOrGeneration') )
#sys.path.append( os.path.join(mainDirectory,'montecarlolearning','TestAndPlot') )
#sys.path.append( os.path.join(mainDirectory,'montecarlolearning','DifferentialML') )
#sys.path.append( os.path.join(mainDirectory,'montecarlolearning','TrainingProcess') )
#sys.path.append( os.path.join(mainDirectory,'montecarlolearning','NeuralNetworkApproximator') )
sys.path.append( os.path.join(mainDirectory,'src','Examples','CumulativeDensitiyFunction/' ))

# Data importation or generation classes
#from BlackScholes import * # not used
from CDF import *
from GBM_5d import *
from GBM import *
from GBM_Differential import *
from DataImporter import *
from Multilevel_GBM import *

# Training and testing functions
#from train_and_test_with_differentials import * # not used
from train_and_test import *
from TrainingOptionEnums import *

# Plot function
from plot_results import *

from Neural_Approximator import *
from TrainingSettings import *
from vanilla_net import *
from backprop import *
from normalize_data import *
from vanilla_graph import *
from diff_training_graph import * # not used
from train import *
from Neural_Approximator import *
from vanilla_net import *
from vanilla_net_biasNeuron import *
from backprop import *
#from vanilla_graph import *
from vanilla_train_one_epoch import *
#from diff_training_graph import *

# Multilevel
from train_and_test_Multilevel import *
from Neural_Approximator_Multilevel  import *
from GBM_Multilevel import *