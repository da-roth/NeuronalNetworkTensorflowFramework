###
### Import packages
###
from normalize_data import *
from vanilla_graph import *
from diff_training_graph import *
from train import *
import tensorflow as tf2
#print("TF version =", tf2.__version__)
# we want TF 2.x
assert tf2.__version__ >= "2.0"
# disable eager execution etc
tf = tf2.compat.v1
tf.disable_eager_execution()

###
### Neural network class
###
class Neural_Approximator():
    ###
    ### Attributes
    ###
    
    # raw data
    x_raw = None
    y_raw = None
    dydx_raw = None
    
    # Tensorflow logics
    graph = None
    session = None
    
    # data and data analytics (initialized in batch normalization)
    x   = None 
    y = None
    x_mean = None
    y_mean = None
    x_std = None
    y_std = None
    dy_dx = None
    lambda_j = None
    
    ###
    ### Constructor / Destructor
    ###
    
    def __init__(self, x_raw = None, y_raw= None, 
                dydx_raw=None):      # derivatives labels, 
        # Raw data input
        self.x_raw = x_raw
        self.y_raw = y_raw
        self.dydx_raw = dydx_raw
                  
    def __del__(self):
        if self.session is not None:
            self.session.close()
        
    ###
    ### Properties
    ###
    
    # Prepare:                    
    # - Normalize data and data analytics (x_mean)
    # - Build graph
    def prepare(self, 
                m, 
                differential,
                hiddenNeurons, 
                hiddenLayers, 
                activationFunctionsHidden,
                activationFunctionOutput = tf.nn.softplus,
                lam=1,
                weight_seed=None):

        # Normalize dataset and cache analytics
        self.x_mean, self.x_std, self.x, self.y_mean, self.y_std, self.y, self.dy_dx, self.lambda_j = \
            normalize_data(self.x_raw, self.y_raw, self.dydx_raw, m)
        
        # Build tensorflow graph        
        self.m, self.n = self.x.shape        
        self.build_graph(differential, lam, hiddenNeurons, hiddenLayers, activationFunctionsHidden, activationFunctionOutput, weight_seed)
        
        
    def storeNewDataAndNormalize(self, x_raw, y_raw, dydx_raw, dataSize):
        
        # Normalize dataset and cache analytics
        self.x_mean, self.x_std, self.x, self.y_mean, self.y_std, self.y, self.dy_dx, self.lambda_j = \
            normalize_data(x_raw, y_raw, dydx_raw, dataSize)
        
        
    # Build graph
    def build_graph(self,
                differential,       # differential or not           
                lam,                # balance cost between values and derivs  
                hiddenNeurons, 
                hiddenLayers,
                activationFunctionsHidden, 
                activationFunctionOutput,
                weight_seed):
        
        # First, deal with tensorflow logic
        if self.session is not None:
            self.session.close()

        #Print neural network settings
        #print('Neural network initialized with the following settings:')
        #print('Neurons per layer: ' + str(hiddenNeurons))
        #print('Amount of hidden layers: ' +str(hiddenLayers))
        #print('Activations functions: ' + str(activationFunctions))
        
        self.graph = tf.Graph()
        
        with self.graph.as_default():
        
            # Build the graph, either vanilla or differential
            self.differential = differential
            
            if not differential:
            # Build vanilla graph through vanilla_graph.py
                
                self.inputs, \
                self.labels, \
                self.predictions, \
                self.derivs_predictions, \
                self.learning_rate, \
                self.loss, \
                self.minimizer \
                = vanilla_training_graph(self.n, hiddenNeurons, hiddenLayers, activationFunctionsHidden, activationFunctionOutput, weight_seed)
                    
            else:
            # Build differential graph through diff_training_graph.py
            
                if self.dy_dx is None:
                    raise Exception("No differential labels for differential training graph")
            
                self.alpha = 1.0 / (1.0 + lam * self.n)
                self.beta = 1.0 - self.alpha
                
                self.inputs, \
                self.labels, \
                self.derivs_labels, \
                self.predictions, \
                self.derivs_predictions, \
                self.learning_rate, \
                self.loss, \
                self.minimizer = diff_training_graph(self.n, hidden_units, \
                                                     hidden_layers, weight_seed, \
                                                     self.alpha, self.beta, self.lambda_j)
        
            # Global initializer
            self.initializer = tf.global_variables_initializer()
            
        # Done building graph
        self.graph.finalize()
        self.session = tf.Session(graph=self.graph)
    
    # Train network by calling train based on train.py
    def train(self,            
              description,
              # training params
              epochs, 
              # one-cycle learning rate schedule
              learning_rate_schedule,
              batches_per_epoch,
              min_batch_size,
              reinit=True, 
              # callback and when to call it
              # we don't use callbacks, but this is very useful, e.g. for debugging
              callback=None,           # arbitrary callable
              callback_epochs=[],
              xTest=None,
              yTest=None              
              ):     # call after what epochs, e.g. [5, 20]
              
        train(description, 
              self, 
              epochs, 
              learning_rate_schedule, 
              batches_per_epoch, 
              min_batch_size,
              reinit, 
              callback, 
              callback_epochs,
              xTest,
              yTest)
     
    # Predict values
    def predict_values(self, x):
        # scale
        x_scaled = (x-self.x_mean) / self.x_std 
        # predict scaled
        y_scaled = self.session.run(self.predictions, feed_dict = {self.inputs: x_scaled})
        # unscale
        y = self.y_mean + self.y_std * y_scaled
        return y

    # Predict values and derivatives
    def predict_values_and_derivs(self, x):
        # scale
        x_scaled = (x-self.x_mean) / self.x_std
        # predict scaled
        y_scaled, dyscaled_dxscaled = self.session.run(
            [self.predictions, self.derivs_predictions], 
            feed_dict = {self.inputs: x_scaled})
        # unscale
        y = self.y_mean + self.y_std * y_scaled
        dydx = self.y_std / self.x_std * dyscaled_dxscaled
        return y, dydx
