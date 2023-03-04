import numpy as np
from scipy.stats import norm

try:
    from TrainingOptionEnums import *
except ModuleNotFoundError:
    #print("")
    from montecarlolearning.TrainingOptionEnums import *

# main class
class CDF:
    
    ###
    ### Attributes
    ###

    # Specific for CDF
    _mean = None
    _vol = None
    _dataSeed = None
    _testSeed = None 

    # Mandatory for train and regressor
    _differential = None
    _trainingMethod = TrainingMethod.GenerateDataDuringTraining
    _inputName = None
    _outputName = None
    _trainingSetSizes = None
    _nTest = None



    @property
    def inputName(self):
        if self._inputName is None:
            raise ValueError("Input name attribute is not set. Please add a value for DataImporter.")
        return self._inputName

    @property
    def outputName(self):
        if self._outputName is None:
            raise ValueError("Output name attribute is not set. Please add a value for DataImporter.")
        return self._outputName

    @property
    def trainingSetSizes(self):
        if self._trainingSetSizes is None:
            raise ValueError("Training set sizes attribute is not set. Please add a value for DataImporter.")
        return self._trainingSetSizes

    @property # Allowed to be None
    def nTest(self):
        return self._nTest

    @property
    def dataSeed(self):
        if self._dataSeed is None:
            raise ValueError("Data seed attribute is not set. Please add a value for DataImporter.")
        return self._dataSeed

    @property
    def testSeed(self):
        if self._testSeed is None:
            raise ValueError("Test seed attribute is not set. Please add a value for DataImporter.")
        return self._testSeed
    
    @property
    def TrainMethod(self):
        return self._trainingMethod
    
    @property
    def Differential(self):
        return self._differential

    ###
    ### Constructor
    ###
    def __init__(self, 
                 mean=0.0,
                 vol=1.0):
        self._differential = False
        self._sep = None
        self._nTest = None
        self._dataSeed = 1 
        self._testSeed = 0
        self.mean = 0.0
        self.vol = 1.0

    def set_inputName(self, inputName):
        self._inputName = inputName

    def set_outputName(self, outputName):
        self._outputName = outputName

    def set_trainingSetSizes(self, trainingSetSizes):
        self._trainingSetSizes = trainingSetSizes

    def set_dataSeed(self, dataSeed):
        self._dataSeed = dataSeed

    def set_testSeed(self, testSeed):
        self._testSeed = testSeed
    
    def set_nTest(self, value):
        self._nTest = value
                        
    # training set: returns CDF for m random inputs
    def trainingSet(self, m, trainSeed = None):
    
        np.random.seed(trainSeed)
        
        # 2 sets of normal returns
        b = 4.01
        a = -4.01
        randomInputs = (b - a) * np.random.random_sample(m) + a
        cdfVector = norm.cdf(randomInputs,self.mean,self.vol)

        return randomInputs.reshape([-1,1]) , cdfVector.reshape([-1,1]), None
    
    # test set: returns a grid of uniform spots 
    def testSet(self, num, testSeed = None):

        np.random.seed(testSeed)

        b = 4.0
        a = -4.0
        testInputs = np.linspace(a, b, num)
        cdfVector = norm.cdf(testInputs,self.mean,self.vol)
        return testInputs.reshape([-1,1]), cdfVector.reshape([-1,1]), None, None
