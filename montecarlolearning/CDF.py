import numpy as np
from scipy.stats import norm

# main class
class CDF:
    
    def __init__(self, 
                 mean=0.0,
                 vol=1.0):
        
        self.mean = mean
        self.vol = vol
                        
    # training set: returns CDF for m random inputs
    def trainingSet(self, m, trainSeed=None):
    
        np.random.seed(trainSeed)
        
        # 2 sets of normal returns
        b = 4.01
        a = -4.01
        randomInputs = (b - a) * np.random.random_sample(m) + a
        cdfVector = norm.cdf(randomInputs,self.mean,self.vol)

        return randomInputs.reshape([-1,1]) , cdfVector.reshape([-1,1]), None
    
    # test set: returns a grid of uniform spots 
    def testSet(self, num, testSeed=None):
        b = 4.0
        a = -4.0
        testInputs = np.linspace(a, b, num)
        cdfVector = norm.cdf(testInputs,self.mean,self.vol)
        return testInputs.reshape([-1,1]), cdfVector.reshape([-1,1]), None, None
