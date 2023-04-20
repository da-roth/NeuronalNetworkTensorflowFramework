#Import train and test data
import pandas as pd

try:
    from TrainingOptionEnums import *
except ModuleNotFoundError:
    #print("")
    from montecarlolearning.TrainingOptionEnums import *

class DataImporter:
    """
    A class for importing and preprocessing training and test data for machine learning models.

    Attributes:
    ----------
    _trainingMethod (TrainingMethod): The method used for generating the training data.
    _differential (bool): A flag indicating whether or not to use differential training.
    _trainTestRatio (float): The ratio of training data to test data.
    _testDataPath (str): The path to the test data file.
    _randomized (bool): Whether or not to randomize the order of the training data.
    _sep (str): The delimiter used in the input data file.
    _dataSeed (int): The random seed used for generating the training data.
    _testSeed (int): The random seed used for generating the test data.
    _path (str): The path to the input data file.
    _inputName (str): The name of the input variable.
    _outputName (str): The name of the output variable.
    _trainingSetSizes (list): A list of training set sizes.
    _nTest (int): The number of test samples.
    _df (DataFrame): The Pandas DataFrame object containing the input data.
    
    Methods:
    -------
    set_path(path, sep=None):
        Set the path to the input data file.
    set_inputName(inputName):
        Set the name of the input variable.
    set_outputName(outputName):
        Set the name of the output variable.
    set_trainTestRatio(trainTestRatio):
        Set the ratio of training data to test data.
    set_testDataPath(testDataPath):
        Set the path to the test data file.
    set_randomized(randomized):
        Set whether or not to randomize the order of the training data.
    set_trainingSetSizes(trainingSetSizes):
        Set a list of training set sizes.
    set_dataSeed(dataSeed):
        Set the random seed used for generating the training data.
    set_testSeed(testSeed):
        Set the random seed used for generating the test data.
    trainingSet(m, trainSeed=None):
        Generate a training set of a given size.
    testSet(num, testSeed=None):
        Generate a test set of a given size.    
    """
    
    ###
    ### Constructor
    ###
    def __init__(self):
        self._trainingMethod = TrainingMethod.Standard
        self._differential = False
        self._trainTestRatio = 0.8
        self._testDataPath = None
        self._randomized = True
        self._sep = None
        self._dataSeed = 1 
        self._testSeed = 0
        
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
    
    @property
    def Differential(self):
        return self._differential
        
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