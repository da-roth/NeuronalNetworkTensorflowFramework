import numpy as np
from scipy.stats import norm

# main class
class GBM:
                            
    # training set: returns CDF for m random inputs
    def trainingSet(self, m, trainSeed=None, approx=False):
    
        #np.random.seed(trainSeed)
           
        # 1. Input definition
        s_0_l=118.0
        s_0_r=120.0
        s_0 = (s_0_r - s_0_l) * np.random.random_sample(m) + s_0_l
        
        # 2. Fixed parameters
        sigma = 0.2
        mu = 0.05
        T = 1.0
        K = 110

        
        # 3. sets of random returns
        h=T/1.0
        z=np.random.normal(0.0,1.0,m)
        
        #piecewise multiply of s= s_0[:] * np.exp((mu-sigma*sigma/2)*h+sigma*np.sqrt(h)*z[:])
        s= np.multiply(s_0[:],np.exp((mu-0.5*sigma*sigma)*h+sigma*np.sqrt(h)*z[:]))
        
        #payoffs=np.exp(-mu * T) * np.maximum(s[:] - K, 0.0)
        d1 = (np.log(s_0[:]/K) + 0.5 * sigma * sigma * T) / sigma / np.sqrt(T)
        d2 = d1[:] - sigma * np.sqrt(T)
        payoffs = s_0[:] * norm.cdf(d1[:]) - np.exp(-mu*T) * K * norm.cdf(d2[:])
        
        return s_0.reshape([-1,1]) , payoffs.reshape([-1,1]), None
    
    # test set: returns a grid of uniform spots 
    def testSet(self, num, testSeed=None):

        # 1. Input definition
        s_0_l=118.0
        s_0_r=120.0
        s_0 = (s_0_r - s_0_l) * np.random.random_sample(num) + s_0_l
        
        # 2. Fixed parameters
        sigma = 0.2
        mu = 0.05
        T = 1.0
        K = 110
        
        # B.S. formula
        d1 = (np.log(s_0[:]/K) + 0.5 * sigma * sigma * T) / sigma / np.sqrt(T)
        d2 = d1[:] - sigma * np.sqrt(T)
        price = s_0[:] * norm.cdf(d1[:]) - np.exp(-mu*T) * K * norm.cdf(d2[:])
        return s_0.reshape([-1,1]), price.reshape([-1,1]), None, None
