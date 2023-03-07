import numpy as np
from scipy.stats import norm

try:
    from TrainingDataGenerator import *
except ModuleNotFoundError:
    #print("")
    from montecarlolearning.TrainingDataGenerator import *
        
# main class
class EuropeanSDESingle(TrainingDataGenerator):
    
    def __init__(self):
        
        # Call the parent class's constructor using super()
        super().__init__()

        # Mandatory 
        self._differential = False
        
        # Training set definition
        self.s_0_trainInterval = [118.0, 120.0]
        self.sigma_trainInterval = [0.1, 0.2]
        self.mu_trainInterval = [0.02, 0.05]
        self.T_trainInterval = [0.9, 1.0]
        self.K_trainInterval = [109.0, 110.0]
        
        # Test set modification: (reducing test interval slightly for better testing)
        self.s_0_h = 0.4
        self.sigma_h = 0.01
        self.mu_h = 0.01  
        self.T_h = 0.01              
        self.K_h = 0.1
        
    def trainingSet(self, m, trainSeed=None, approx=False):  
        #np.random.seed(trainSeed) 
        # 1. Draw parameter samples for training
        s_0 = (self.s_0_trainInterval[1] - self.s_0_trainInterval[0]) * np.random.random_sample(m) + self.s_0_trainInterval[0]
        sigma = (self.sigma_trainInterval[1] - self.sigma_trainInterval[0]) * np.random.random_sample(m) + self.sigma_trainInterval[0]
        mu = (self.mu_trainInterval[1] - self.mu_trainInterval[0]) * np.random.random_sample(m) + self.mu_trainInterval[0]
        T = (self.T_trainInterval[1] - self.T_trainInterval[0]) * np.random.random_sample(m) + self.T_trainInterval[0]
        K = (self.K_trainInterval[1] - self.K_trainInterval[0]) * np.random.random_sample(m) + self.K_trainInterval[0]

        # 2. Compute paths
        h=T[:]/1.0
        z=np.random.normal(0.0, 1.0, m)
        s= s_0[:] + mu[:] *s_0[:] * h[:] +sigma[:] * s_0[:] *np.sqrt(h[:])*z[:] + 0.5 *sigma[:] *s_0[:] *sigma[:] * ((np.sqrt(h[:])*z[:])**2-h[:]) 
        
        # 3. Calculate and return payoffs
        payoffs=np.exp(-mu[:] * T[:])* np.maximum(s[:] - K[:], 0.)
        return np.array([s_0,sigma,mu,T,K]).reshape([-1,5]) , payoffs.reshape([-1,1]), None
     
    def testSet(self, num, testSeed=None):

        # 0. Test interval definition
        s_0_testInterval = [self.s_0_trainInterval[0]+self.s_0_h, self.s_0_trainInterval[1]-self.s_0_h]
        sigma_testInterval = [self.sigma_trainInterval[0]+self.sigma_h, self.sigma_trainInterval[1]-self.sigma_h]
        mu_testInterval = [self.mu_trainInterval[0]+self.mu_h, self.mu_trainInterval[1]-self.mu_h]
        T_testInterval = [self.T_trainInterval[0]+self.T_h, self.T_trainInterval[1]-self.T_h]
        K_testInterval = [self.K_trainInterval[0]+self.K_h, self.K_trainInterval[1]-self.K_h]

        # 1. Draw parameter samples for test
        s_0 = (s_0_testInterval[1] - s_0_testInterval[0]) * np.random.random_sample(num) + s_0_testInterval[0]
        sigma = (sigma_testInterval[1] - sigma_testInterval[0]) * np.random.random_sample(num) + sigma_testInterval[0]
        mu = (mu_testInterval[1] - mu_testInterval[0]) * np.random.random_sample(num) + mu_testInterval[0]
        T = (T_testInterval[1] - T_testInterval[0]) * np.random.random_sample(num) + T_testInterval[0]
        K = (K_testInterval[1] - K_testInterval[0]) * np.random.random_sample(num) + K_testInterval[0]
        
        # B.S. formula
        d1 = (np.log(s_0[:]/K[:]) + 0.5 * sigma[:] * sigma[:] * T[:]) / sigma[:] / np.sqrt(T[:])
        d2 = d1[:] - sigma[:] * np.sqrt(T[:])
        price = s_0[:] * norm.cdf(d1[:]) - np.exp(-mu[:] * T[:]) * K[:] * norm.cdf(d2[:])
        return np.array([s_0,sigma,mu,T,K]).reshape([-1,5]), price.reshape([-1,1]), None, None
    
#unused helper
# # helper analytics    
# #European option
# def phi(x,sigma,mu,T,K, axis=1):
#     payoffcoarse=np.exp(-mu * T)* np.maximum(x - K, 0.)
#     return payoffcoarse
# #Milstein scheme
# def sde_body(idx, s, sigma,mu,T,K, samples, batch_size, dtype, N): 
#     h=T/N
#     z=np.random.normal(size=[m, 2])
#     s=s + mu *s * h +sigma * s *np.sqrt(h)*z[:,1] + 0.5 *sigma *s *sigma * ((np.sqrt(h)*z[:,2])**2-h)    
#     return np.add(idx, 1), s, sigma,mu,T,K
# #Monte Carlo loop                 
# def mc_body(idx, p, N, mc_samples_trainInterval[1]ef, loop_var_mc):
#     _, _x, _sigma,_mu,_T,_K = np.while_trainInterval[0]oop(lambda _idx, s, sigma,mu,T,K: _idx < N,
#                             lambda _idx, s, sigma,mu,T,K: sde_body(_idx, s, sigma,mu,T,K,
#                                                     mc_samples_trainInterval[1]ef),
#                                                     loop_var_mc)
#     return idx + 1, p + np.reduce_mean(phi(_x,_sigma,_mu,_T,_K, 2), axis=0)