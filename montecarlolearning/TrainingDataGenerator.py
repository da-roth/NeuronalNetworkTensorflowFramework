import numpy as np
from scipy.stats import norm

try:
    from TrainingOptionEnums import *
except ModuleNotFoundError:
    #print("")
    from montecarlolearning.TrainingOptionEnums import *

# main class
class TrainingDataGenerator:
    


    ###
    ### Constructor
    ###
    def __init__(self):
        # Mandatory for all
        self._differential = None

        # Mandatory for all but with defaults:
        self._trainingMethod = TrainingMethod.GenerateDataDuringTraining
        self._inputName = 'x'
        self._outputName = 'y'
        self._dataSeed = 1
        self._testSeed = 1

    ###
    ### Getter and setter
    ###

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
    def TrainMethod(self):
        if self._trainingMethod is None:
            raise ValueError("Test method attribute is not set. Please set it in the TrainingDataOrGenerator subclass")
        return self._trainingMethod
    
    @property
    def Differential(self):
        if self._differential is None:
            raise ValueError("Differential attribute is not set. Please add a value for DataImporter.")
        return self._differential
           

    def set_nTest(self, value):
        self._nTest = value

    def set_inputName(self, inputName):
        self._inputName = inputName

    def set_outputName(self, outputName):
        self._outputName = outputName

    def set_dataSeed(self, dataSeed):
        self._dataSeed = dataSeed

    def set_testSeed(self, testSeed):
        self._testSeed = testSeed
         
    # training set: returns CDF for m random inputs
    def trainingSet(self, m, trainSeed = None):
        raise NotImplementedError("Subclass must implement abstract method")
    
    # test set: returns a grid of uniform spots 
    def testSet(self, num, testSeed = None):
        raise NotImplementedError("Subclass must implement abstract method")
