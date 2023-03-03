#Import train and test data
import pandas as pd

class DataImporter:
        
    ###
    ### Attributes
    ###
    path = None
    inputName = None
    outputName = None
    trainTestRatio = None
    testDataPath = None
    randomized = None
    sep = None

    df = None

    ###
    ### Constructor
    ###
    def __init__(self):
        self.trainTestRatio = 0.8
        self.testDataPath = None
        self.randomized = True
        self.sep = None
        
    def set_path(self, path, sep = None):
        self.path = path
        self.sep = sep
        self.df = pd.read_csv(path,sep)
        self.df.head()

    def set_inputName(self, inputName):
        self.inputName = inputName
        
    def set_outputName(self, outputName):
        self.outputName = outputName
        
    def set_trainTestRatio(self, trainTestRatio):
        self.trainTestRatio = trainTestRatio
        
    def set_testDataPath(self, testDataPath):
        self.testDataPath = testDataPath
        
    def set_randomized(self, randomized):
        self.randomized = randomized
        
    def set_sep(self, sep):
        self.sep = sep

    ###
    ### Methods
    ### 
    def trainingSet(self,m, trainSeed = None):
        if not self.testDataPath:
            if (trainSeed == None):
                self.train_df = self.df.sample(frac=self.trainTestRatio, random_state=1)
            else:
                self.train_df = self.df.sample(frac=self.trainTestRatio, random_state=trainSeed)
            xTrain = self.train_df.drop(self.outputName,axis=1)
            yTrain = self.train_df.drop(self.inputName,axis=1)
            return xTrain.values, yTrain.values, None
                
        else:
            if self.randomized:
                train_df = self.df.sample(frac=1.0, random_state=1)
                xTrain = train_df.drop(self.outputName,axis=1)
                yTrain = train_df.drop(self.inputName,axis=1)
                return xTrain.values, yTrain.values, None
            else:
                train_df = self.df
                xTrain = train_df.drop(self.outputName,axis=1)
                yTrain = train_df.drop(self.inputName,axis=1)
                return xTrain.values, yTrain.values, None
                
            
            
          
    def testSet(self, num, testSeed=None):
        if not self.testDataPath:
            self.test_df = self.df.drop(self.train_df.index)
            xTest = self.test_df.drop(self.outputName,axis=1)
            yTest = self.test_df.drop(self.inputName,axis=1)
            return xTest.values, yTest.values, None, None
        else:
            dfTest = pd.read_csv(self.testDataPath) 
            dfTest.head()
            dfTest = dfTest.sample(frac=0.1, random_state=1)
            xTest = dfTest.drop(self.outputName,axis=1)
            yTest = dfTest.drop(self.inputName,axis=1)
            return xTest.values, yTest.values, None, None