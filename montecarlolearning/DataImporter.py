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

    ###
    ### Constructor
    ###
    def __init__(self, 
                 path,
                 inputName,
                 outputName,
                 trainTestRatio=0.8,
                 testDataPath = None,
                 randomized = True,
                 sep = None,
                 ):
        
        self.path = path
        self.inputName = inputName
        self.outputName = outputName
        self.df = pd.read_csv(path,sep) 
        self.df.head()
        self.trainTestRatio = trainTestRatio
        self.testDataPath = testDataPath
        self.randomized = randomized
        

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