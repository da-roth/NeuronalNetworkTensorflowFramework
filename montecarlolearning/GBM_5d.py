import numpy as np
from scipy.stats import norm
import scipy.stats as stats
from enum import Enum 

try:
    from TrainingDataGenerator import *
except ModuleNotFoundError:
    #print("")
    from montecarlolearning.TrainingDataGenerator import *

class GBM_5d_Case(Enum):
    Standard = 1                    # European call (Strike 70) and one MC sample per input
    VarianceReduced = 2             # European call (Strike 70) and one MC sample with variance reduction (importance sampling forcing to stay above K)
    ClosedSolution = 3              # European call with closed solution as output
    ClosedSolutionAddtiveNoise = 4  # European call with closed solution plus additive noise as output

# helper analytics    
#European _option
def phi(x,sigma,mu,T,K, axis=1):
    payoffcoarse=np.exp(-mu * T)* np.maximum(x - K, 0.)
    return payoffcoarse
#Milstein scheme
def sde_body(idx, s, sigma,mu,T,K, samples, batch_size, dtype, N): 
    h=T/N
    z=np.random.normal(size=[m, 2])
    s=s + mu *s * h +sigma * s *np.sqrt(h)*z[:,1] + 0.5 *sigma *s *sigma * ((np.sqrt(h)*z[:,2])**2-h)    
    return np.add(idx, 1), s, sigma,mu,T,K
#Monte Carlo loop                 
def mc_body(idx, p, N, mc_samples_ref, loop_var_mc):
    _, _x, _sigma,_mu,_T,_K = np.while_loop(lambda _idx, s, sigma,mu,T,K: _idx < N,
                            lambda _idx, s, sigma,mu,T,K: sde_body(_idx, s, sigma,mu,T,K,
                                                    mc_samples_ref),
                                                    loop_var_mc)
    return idx + 1, p + np.reduce_mean(phi(_x,_sigma,_mu,_T,_K, 2), axis=0)
    
# main class
class GBM_5d(TrainingDataGenerator):

    ###
    ### Attributes
    ###
    _opt = None
    _noiseVariance = None
    
    def __init__(self, 
                opt = GBM_5d_Case.Standard,
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
    
        #np.random.seed(trainSeed)
           
        # 1. Input definition
        s_0_l=110.0
        s_0_r=120.0
        s_0 = (s_0_r - s_0_l) * np.random.random_sample(m) + s_0_l
        sigma_l=0.1
        sigma_r=0.2
        sigma = (sigma_r - sigma_l) * np.random.random_sample(m) + sigma_l
        mu_l=0.02
        mu_r=0.05
        mu = (mu_r - mu_l) * np.random.random_sample(m) + mu_l
        T_l=0.9
        T_r=1.0
        T = (T_r - T_l) * np.random.random_sample(m) + T_l
        K_l=89.0
        K_r=90.0
        K = (K_r - K_l) * np.random.random_sample(m) + K_l

        
        if (self._opt == GBM_5d_Case.ClosedSolutionAddtiveNoise):
            d1 = (np.log(s_0[:]/K[:]) + 0.5 * sigma[:] * sigma[:] * T[:]) / sigma[:] / np.sqrt(T[:])
            d2 = d1[:] - sigma[:] * np.sqrt(T[:])
            z=np.random.normal(0.0, self._noiseVariance, m)
            noisedPrice = s_0[:] * norm.cdf(d1[:]) - np.exp(-mu[:]*T[:]) * K[:] * norm.cdf(d2[:]) + z[:]
            return np.stack((s_0,sigma,mu,T,K),axis=1), noisedPrice.reshape([-1,1]), None
        
        elif (self._opt == GBM_5d_Case.ClosedSolution):
            d1 = (np.log(s_0[:]/K[:]) + 0.5 * sigma[:] * sigma[:] * T[:]) / sigma[:] / np.sqrt(T[:])
            d2 = d1[:] - sigma[:] * np.sqrt(T[:])
            price = s_0[:] * norm.cdf(d1[:]) - np.exp(-mu[:] *T[:] ) * K[:] * norm.cdf(d2[:])
            return np.stack((s_0,sigma,mu,T,K),axis=1), price.reshape([-1,1]), None

        elif (self._opt == GBM_5d_Case.VarianceReduced):
            #3. sets of random returns
            h=T[:]/1.0
            #z=np.random.normal(0.0,1.0,m)
            p = norm.cdf((np.log(K[:] /s_0[:])-(mu[:] -0.5*sigma[:] *sigma[:] *T[:] ))/(sigma[:] *np.sqrt(T[:] )))
            u = np.random.uniform(low=0.0, high=1.0, size=m)
            z = norm.ppf((1-p[:])*u[:]+p[:])

            #piecewise multiply of s= s_0[:] * np.exp((mu-sigma*sigma/2)*h+sigma*np.sqrt(h)*z[:])
            s= np.multiply(s_0[:],np.exp((mu[:] -0.5*sigma[:] *sigma[:] )*h+sigma[:] *np.sqrt(h[:] )*z[:]))
            payoffs=np.exp(-mu[:]  * T[:] ) * (s[:]-K[:] ) * (1-p[:])
            return np.stack((s_0,sigma,mu,T,K),axis=1), payoffs.reshape([-1,1]), None
        
        else: #(self._opt == GBM_5d_Case.Standard):
            #3. sets of random returns
            h=T[:] /1.0
            z=np.random.normal(0.0,1.0,m)
            #piecewise multiply of s= s_0[:] * np.exp((mu-sigma*sigma/2)*h+sigma*np.sqrt(h)*z[:])
            s= np.multiply(s_0[:],np.exp((mu[:] -0.5*sigma[:] *sigma[:] )*h[:] +sigma[:] *np.sqrt(h[:] )*z[:]))
            payoffs=np.exp(-mu[:]  * T[:] ) * np.maximum(s[:] - K[:] , 0.0)
            return np.stack((s_0,sigma,mu,T,K),axis=1), payoffs.reshape([-1,1]), None
    
    # test set: returns a grid of uniform spots 
    def testSet(self, num, testSeed=None):

        # 1. Input definition
        s_0_l=110.4
        s_0_r=119.6
        s_0 = (s_0_r - s_0_l) * np.random.random_sample(num) + s_0_l
        sigma_l=0.11
        sigma_r=0.19
        sigma = (sigma_r - sigma_l) * np.random.random_sample(num) + sigma_l
        mu_l=0.03
        mu_r=0.04
        mu = (mu_r - mu_l) * np.random.random_sample(num) + mu_l
        T_l=0.91
        T_r=0.99
        T = (T_r - T_l) * np.random.random_sample(num) + T_l
        K_l=89.1
        K_r=89.9
        K = (K_r - K_l) * np.random.random_sample(num) + K_l
        
        # B.S. formula
        d1 = (np.log(s_0[:]/K[:]) + 0.5 * sigma[:] * sigma[:] * T[:]) / sigma[:] / np.sqrt(T[:])
        d2 = d1[:] - sigma[:] * np.sqrt(T[:])
        price = s_0[:] * norm.cdf(d1[:]) - np.exp(-mu[:] * T[:]) * K[:] * norm.cdf(d2[:])
        return np.stack((s_0,sigma,mu,T,K),axis=1), price.reshape([-1,1]), None, None
