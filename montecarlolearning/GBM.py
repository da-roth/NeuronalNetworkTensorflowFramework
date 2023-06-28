import numpy as np
from scipy.stats import norm
import scipy.stats as stats
from enum import Enum 

try:
    from TrainingDataGenerator import *
except ModuleNotFoundError:
    #print("")
    from montecarlolearning.TrainingDataGenerator import *


class GBM_Case(Enum):
    Standard = 1                    # European call (Strike 70) and one MC sample per input
    VarianceReduced = 2             # European call (Strike 70) and one MC sample with variance reduction (importance sampling forcing to stay above K)
    ClosedSolution = 3              # European call with closed solution as output
    ClosedSolutionAddtiveNoise = 4  # European call with closed solution plus additive noise as output
    
    
# main class
class GBM(TrainingDataGenerator):
    
    ###
    ### Constructor
    ###
    def __init__(self, 
                opt = GBM_Case.Standard,
                noiseVariance = 0.1):
        
        # Call the parent class's constructor using super()
        super().__init__()

        # Mandatory 
        self._differential = False
        
        self._opt = opt
        self._noiseVariance = noiseVariance

    def set_noiseVariance(self, inputName):
        self._noiseVariance = inputName

    def set_trainingCase(self, inputName):
        self._opt = inputName
                            
    # training set: returns CDF for m random inputs
    def trainingSet(self, m, trainSeed=None, approx=False):
    
        np.random.seed(trainSeed)
           
        # 1. Input definition
        s_0_l=110.0
        s_0_r=120.0
        s_0 = (s_0_r - s_0_l) * np.random.random_sample(m) + s_0_l
        
        # 2. Fixed parameters
        sigma = 0.2
        mu = 0.05
        T = 1.0
        K = 110


        if (self._opt == GBM_Case.ClosedSolutionAddtiveNoise):
            d1 = (np.log(s_0[:]/K) + (mu + 0.5 * sigma * sigma) * T) / sigma / np.sqrt(T)
            d2 = d1[:] - sigma * np.sqrt(T)
            z=np.random.normal(0.0, self._noiseVariance, m)
            noisedPrice = s_0[:] * norm.cdf(d1[:]) - np.exp(-mu*T) * K * norm.cdf(d2[:]) + z[:]
            return s_0.reshape([-1,1]), noisedPrice.reshape([-1,1]), None
        
        elif (self._opt == GBM_Case.ClosedSolution):
            d1 = (np.log(s_0[:]/K) +(mu + 0.5 * sigma * sigma) * T) / sigma / np.sqrt(T)
            d2 = d1[:] - sigma * np.sqrt(T)
            price = s_0[:] * norm.cdf(d1[:]) - np.exp(-mu*T) * K * norm.cdf(d2[:])
            return s_0.reshape([-1,1]), price.reshape([-1,1]), None

        elif (self._opt == GBM_Case.VarianceReduced):
            #3. sets of random returns
            h=T/1.0
            #z=np.random.normal(0.0,1.0,m)
            p = norm.cdf((np.log(K/s_0[:])-(mu-0.5*sigma*sigma*T))/(sigma*np.sqrt(T)))
            u = np.random.uniform(low=0.0, high=1.0, size=m)
            z = norm.ppf((1-p[:])*u[:]+p[:])

            #piecewise multiply of s= s_0[:] * np.exp((mu-sigma*sigma/2)*h+sigma*np.sqrt(h)*z[:])
            s= np.multiply(s_0[:],np.exp((mu-0.5*sigma*sigma)*h+sigma*np.sqrt(h)*z[:]))
            payoffs=np.exp(-mu * T) * (s[:]-K) * (1-p[:])
            return s_0.reshape([-1,1]) , payoffs.reshape([-1,1]), None
        
        else: #(self._opt == GBM_Case.Standard):
            #3. sets of random returns
            h=T/1.0
            z=np.random.normal(0.0,1.0,m)
            #piecewise multiply of s= s_0[:] * np.exp((mu-sigma*sigma/2)*h+sigma*np.sqrt(h)*z[:])
            s= np.multiply(s_0[:],np.exp((mu-0.5*sigma*sigma)*h+sigma*np.sqrt(h)*z[:]))
            payoffs=np.exp(-mu * T) * np.maximum(s[:] - K, 0.0)
            return s_0.reshape([-1,1]) , payoffs.reshape([-1,1]), None
        
    
    # test set: returns a grid of uniform spots 
    def testSet(self, num, testSeed=None):

        np.random.seed(1)
        # 1. Input definition
        s_0_l=110.0
        s_0_r=120.0
        s_0 = (s_0_r - s_0_l) * np.random.random_sample(num) + s_0_l
        
        # 2. Fixed parameters
        sigma = 0.2
        mu = 0.05
        T = 1.0
        K = 110
        
        # B.S. formula
        d1 = (np.log(s_0[:]/K) + (mu + 0.5 * sigma * sigma) * T) / sigma / np.sqrt(T)
        d2 = d1[:] - sigma * np.sqrt(T)
        price = s_0[:] * norm.cdf(d1[:]) - np.exp(-mu*T) * K * norm.cdf(d2[:])
        return s_0.reshape([-1,1]), price.reshape([-1,1]), None, None
