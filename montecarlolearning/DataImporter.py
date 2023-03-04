#Import train and test data
import pandas as pd

try:
    from TrainingOptionEnums import *
except ModuleNotFoundError:
    #print("")
    from montecarlolearning.TrainingOptionEnums import *

class DataImporter:
        
    ###
    ### Attributes
    ###

    @property
    def path(self):
        if self._path is None:
           raise ValueError("Path attribute is not set. Please add a value for DataImporter.")
        return self._path
    
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
    def trainTestRatio(self):
        if self._trainTestRatio is None:
            raise ValueError("Train-test ratio attribute is not set. Please add a value for DataImporter.")
        return self._trainTestRatio

    @property
    def testDataPath(self): # Allowed to be None
        return self._testDataPath

    @property
    def trainingSetSizes(self):
        if self._trainingSetSizes is None:
            raise ValueError("Training set sizes attribute is not set. Please add a value for DataImporter.")
        return self._trainingSetSizes

    @property # Allowed to be None
    def nTest(self):
        return self._nTest

    @property
    def df(self):
        if self._df is None:
            raise ValueError("DataFrame attribute is not set. Please add a value for DataImporter.")
        return self._df

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
    def randomized(self):
        if self._randomized is None:
            raise ValueError("Randomized is not set. Please add a value for DataImporter.")
        return self._randomized
    
    @property
    def TrainMethod(self):
        return self._trainingMethod
    
    # Mandatory
    _trainingMethod = TrainingMethod.Standard
    _path = None
    _inputName = None
    _outputName = None
    _trainTestRatio = None
    _testDataPath = None
    _trainingSetSizes = None
    _nTest = None
    _df = None
    _dataSeed = None
    _testSeed = None 

    # Optional 
    _randomized = None

    # Training set that is created
    _train_df = None


    ###
    ### Constructor
    ###
    def __init__(self):
        self._trainTestRatio = 0.8
        self._testDataPath = None
        self._randomized = True
        self._sep = None
        self._nTest = None
        self._dataSeed = 1 
        self._testSeed = 0
        
    def set_path(self, path, sep=None):
        self._path = path
        self._sep = sep
        self._df = pd.read_csv(path, sep)
        self._df.head()

    def set_inputName(self, inputName):
        self._inputName = inputName

    def set_outputName(self, outputName):
        self._outputName = outputName

    def set_trainTestRatio(self, trainTestRatio):
        self._trainTestRatio = trainTestRatio

    def set_testDataPath(self, testDataPath):
        self._testDataPath = testDataPath

    def set_randomized(self, randomized):
        self._randomized = randomized

    def set_trainingSetSizes(self, trainingSetSizes):
        self._trainingSetSizes = trainingSetSizes

    def set_dataSeed(self, dataSeed):
        self._dataSeed = dataSeed

    def set_testSeed(self, testSeed):
        self._testSeed = testSeed

    ###
    ### Methods
    ### 
    def trainingSet(self,m, trainSeed = None):
        if not self.testDataPath:
            if self.randomized:
                if (trainSeed == None):
                    self._train_df = self._df.sample(frac=self._trainTestRatio, random_state=1)
                else:
                    self._train_df = self._df.sample(frac=self.trainTestRatio, random_state=trainSeed)
                xTrain = self._train_df.drop(self._outputName,axis=1)
                yTrain = self._train_df.drop(self._inputName,axis=1)
                return xTrain.values, yTrain.values, None
            else:
                self._train_df = self._df
                xTrain = self._train_df.drop(self._outputName,axis=1)
                yTrain = self._train_df.drop(self._inputName,axis=1)
                return xTrain.values, yTrain.values, None
                
        else:
            if self.randomized:
                self._train_df = self._df.sample(frac=1.0, random_state=1)
                xTrain = self._train_df.drop(self._outputName,axis=1)
                yTrain = self._train_df.drop(self._inputName,axis=1)
                return xTrain.values, yTrain.values, None
            else:
                self._train_df = self._df
                xTrain = self._train_df.drop(self._outputName,axis=1)
                yTrain = self._train_df.drop(self._inputName,axis=1)
                return xTrain.values, yTrain.values, None
                
            
            
          
    def testSet(self, num, testSeed=None):
        if not self.testDataPath:
            test_df = self._df.sample(frac=0.1, random_state=1)
            xTest = test_df.drop(self.outputName,axis=1)
            yTest = test_df.drop(self.inputName,axis=1)
            return xTest.values, yTest.values, None, None
        else:
            dfTest = pd.read_csv(self._testDataPath) 
            dfTest.head()
            dfTest = dfTest.sample(frac=0.1, random_state=1)
            xTest = dfTest.drop(self._outputName,axis=1)
            yTest = dfTest.drop(self._inputName,axis=1)
            return xTest.values, yTest.values, None, None