###
### Import packages
###
try:
    from normalize_data import *
    from vanilla_graph import *
    from diff_training_graph import *
    from train import *
except:
    from montecarlolearning.normalize_data import *
    from montecarlolearning.vanilla_graph import *
    from montecarlolearning.diff_training_graph import *
    from montecarlolearning.train import *
    
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

    # Store network architecture to build and reset graph      
    _lam= None               # balance cost between values and derivs  
    _hiddenNeurons= None 
    _hiddenLayers= None 
    _activationFunctionsHidden= None  
    _activationFunctionOutput= None 
    _weight_seed= None 
    _biasNeuron= None 

    # Store data Generator, needed for data normalization framework and input/output dimensions
    _Generator = None
    
    # Setter for data Generator
    def set_Generator(self, Generator):
        self._Generator = Generator
        
    @property
    def Generator(self):
        return self._Generator

    # Setter for lam
    def set_lam(self, lam):
        self._lam = lam

    # Setter for hiddenNeurons
    def set_hiddenNeurons(self, hiddenNeurons):
        self._hiddenNeurons = hiddenNeurons

    # Setter for hiddenLayers
    def set_hiddenLayers(self, hiddenLayers):
        self._hiddenLayers = hiddenLayers

    # Setter for activationFunctionsHidden
    def set_activationFunctionsHidden(self, activationFunctionsHidden):
        self._activationFunctionsHidden = activationFunctionsHidden

    # Setter for activationFunctionOutput
    def set_activationFunctionOutput(self, activationFunctionOutput):
        self._activationFunctionOutput = activationFunctionOutput

    # Setter for weight_seed
    def set_weight_seed(self, weight_seed):
        self._weight_seed = weight_seed

    # Setter for biasNeuron
    def set_biasNeuron(self, biasNeuron):
        self._biasNeuron = biasNeuron

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
    
    def __init__(self):         
        self._lam= 1               
        self._hiddenNeurons= 20 
        self._hiddenLayers= 2 
        self._activationFunctionsHidden= tf.nn.relu
        self._activationFunctionOutput= tf.nn.relu 
        self._biasNeuron= False 
        self._weight_seed = 1
        #self.graph = tf.Graph()
                  
    def __del__(self):
        if self.session is not None:
            self.session.close()
        
    ###
    ### Initialize or reset Graph
    ###
    def initializeAndResetGraph(self): 
        self.graph = tf.Graph()
        print(self.graph)

    ###
    ### Properties
    ###
    def initializeData(self, x_raw = None, y_raw= None,dydx_raw=None): 
        # Raw data input
        self.x_raw = x_raw
        self.y_raw = y_raw
        self.dydx_raw = dydx_raw
        
    # Prepare:                    
    # - Normalize data and data analytics (x_mean)
    # - Build graph
    def prepare(self, dataSize, nTest = None):
        # Normalize dataset and cache analytics
        self.x_mean, self.x_std, self.x, self.y_mean, self.y_std, self.y, self.dy_dx, self.lambda_j = \
            normalize_data(self.x_raw, self.y_raw, self.dydx_raw, dataSize)
        
        # Build tensorflow graph        
        self.m, self.n = self.x.shape 
        if nTest == None:
            self.nTest = self.n     
        else:
            self.nTest = nTest
        self.build_graph(self._lam, self._hiddenNeurons, self._hiddenLayers, self._activationFunctionsHidden, self._activationFunctionOutput,self._weight_seed, self._biasNeuron)

        
        
        
    def storeNewDataAndNormalize(self, x_raw, y_raw, dydx_raw, dataSize):
        
        self.x_raw = x_raw
        self.y_raw = y_raw
        self.dydx_raw = dydx_raw
        self.x_mean, self.x_std, self.x, self.y_mean, self.y_std, self.y, self.dy_dx, self.lambda_j = \
            normalize_data(self.x_raw, self.y_raw, self.dydx_raw, dataSize)
        # Normalize dataset and cache analytics
        #self.x, self.y, self.x, self.dy_dx, self.lambda_j = \
        #    normalize_data_NewData(self.x_raw, self.y_raw, self.x_mean, self.x_std, self.y_mean, self.y_std, self.dydx_raw, dataSize)
        
        
    # Build graph
    def build_graph(self,       # differential or not           
                lam,                # balance cost between values and derivs  
                hiddenNeurons, 
                hiddenLayers,
                activationFunctionsHidden, 
                activationFunctionOutput,
                weight_seed,
                biasNeuron):
        
        # First, deal with tensorflow logic
        if self.session is not None:
            self.session.close()

        #Print neural network settings
        #print('Neural network initialized with the following settings:')
        #print('Neurons per layer: ' + str(hiddenNeurons))
        #print('Amount of hidden layers: ' +str(hiddenLayers))
        #print('Activations functions: ' + str(activationFunctions))
                
        with self.graph.as_default():
            
            if not self._Generator.Differential:
            # Build vanilla graph through vanilla_graph.py
                if isinstance(self.x_raw, tf.Tensor):
                    self.inputs, \
                    self.labels, \
                    self.predictions, \
                    self.derivs_predictions, \
                    self.learning_rate, \
                    self.loss, \
                    self.minimizer, \
                    self.isTraining, \
                    self.predictionsTest \
                    = vanilla_training_graph(self.n, self.nTest, hiddenNeurons, hiddenLayers, activationFunctionsHidden, activationFunctionOutput, weight_seed, biasNeuron, self.x_raw, self.y_raw)
                else:
                    self.inputs, \
                    self.labels, \
                    self.predictions, \
                    self.derivs_predictions, \
                    self.learning_rate, \
                    self.loss, \
                    self.minimizer, \
                    self.isTraining \
                    = vanilla_training_graph(self.n, self.nTest, hiddenNeurons, hiddenLayers, activationFunctionsHidden, activationFunctionOutput, weight_seed, biasNeuron)
                    
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
                self.minimizer , \
                self.isTraining, \
                self.predictionsTest \
                    = diff_training_graph(self.n, self.nTest, hidden_units, \
                                                hidden_layers, weight_seed, \
                                                self.alpha, self.beta, self.lambda_j)
        
            # Global initializer
            self.initializer = tf.global_variables_initializer()
            
        # Done building graph
        #self.graph.finalize()
        self.session = tf.Session(graph=self.graph)
    
    # Train network by calling train based on train.py
    def train(self,            
              description,
              TrainingSettings,
              reinit=True, # reset weights and biases before training
              # callback and when to call it
              # we don't use callbacks, but this is very useful, e.g. for debugging
              callback=None,           # arbitrary callable
              callback_epochs=[],
              xTest=None,
              yTest=None              
              ):     # call after what epochs, e.g. [5, 20]
              
        train(description, 
              self, 
              TrainingSettings,
              reinit, 
              callback, 
              callback_epochs,
              xTest,
              yTest)
     
    # Predict values
    def predict_values(self, x, isTraining = True):
        # scale
        #x_scaled = (x-self.x_mean) / self.x_std 
        # predict scaled
        if isinstance(self.x_raw, tf.Tensor):
            x_scaled = (x-self.session.run(self.x_mean)) / self.session.run(self.x_std)
            if isTraining:
                y_scaled = self.session.run(self.predictions, feed_dict = {self.inputs: x_scaled, self.isTraining: isTraining })
            else:
                y_scaled = self.session.run(self.predictions, feed_dict = {self.predictionsTest: x_scaled, self.isTraining: isTraining })
        else:
            x_scaled = (x-self.x_mean) / self.x_std 
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
            feed_dict = {self.inputs: x_scaled, self.isTraining: True})
        # unscale
        y = self.y_mean + self.y_std * y_scaled
        dydx = self.y_std / self.x_std * dyscaled_dxscaled
        return y, dydx
