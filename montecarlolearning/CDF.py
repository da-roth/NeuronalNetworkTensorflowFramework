import numpy as np
from scipy.stats import norm

try:
    from TrainingDataGenerator import *
except ModuleNotFoundError:
    #print("")
    from montecarlolearning.TrainingDataGenerator import *

# main class
class CDF(TrainingDataGenerator):
    """
    A class for generating training and test data for the cumulative distribution function (CDF).

    Attributes:
    ----------
    _lam (float): Balance cost between values and derivatives.
    _hiddenNeurons (int): Number of neurons in each hidden layer.
    _hiddenLayers (int): Number of hidden layers.
    _activationFunctionsHidden: Activation function used in hidden layers.
    _activationFunctionOutput: Activation function used in the output layer.
    _weight_seed (int): Random seed for initializing weights.

    Methods:
    -------
    trainingSet(m, trainSeed = None):
        Generates a training set of a given size.
    testSet(num, testSeed = None):
        Generates a test set of a given size.
    """
    
    ###
    ### Constructor
    ###
    def __init__(self):

        # Call the parent class's constructor using super()
        super().__init__()

        # Mandatory 
        self._differential = False

        # Specific for CDF_
        self._mean = 0.0
        self._vol = 1.0

    # training set: returns CDF for m random inputs
    def trainingSet(self, m, trainSeed = None):
    
        np.random.seed(trainSeed)
        
        # 2 sets of normal returns
        b = 4.01
        a = -4.01
        randomInputs = (b - a) * np.random.random_sample(m) + a
        cdfVector = norm.cdf(randomInputs,self._mean,self._vol)

        return randomInputs.reshape([-1,1]) , cdfVector.reshape([-1,1]), None
    
    # test set: returns a grid of uniform spots 
    def testSet(self, num, testSeed = None):

        np.random.seed(testSeed)

        b = 4.0
        a = -4.0
        testInputs = np.linspace(a, b, num)
        cdfVector = norm.cdf(testInputs,self._mean,self._vol)
        return testInputs.reshape([-1,1]), cdfVector.reshape([-1,1]), None, None
