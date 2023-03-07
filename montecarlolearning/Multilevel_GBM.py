import numpy as np
from scipy.stats import norm

try:
    from TrainingDataGenerator import *
except ModuleNotFoundError:
    #print("")
    from montecarlolearning.TrainingDataGenerator import *
        
class Multilevel_Train_Case(Enum):
    BS_Solution = 1          # B.S. formula
    GBM_Path_Solution = 2    # GBM path closed solution
    Milstein = 3             # Milstein scheme

class Multilevel_Train_Dimension(Enum):
    one = 1
    five = 2

# main class
class Multilevel_GBM(TrainingDataGenerator):
    
    def __init__(self, opt=Multilevel_Train_Case.BS_Solution, steps = 1, dim = Multilevel_Train_Dimension.one):
        
        # Call the parent class's constructor using super()
        super().__init__()

        # Mandatory 
        self._differential = False
        
        # Multilevel GBM specific
        self._opt = opt # Case
        self._steps = steps     # discretization steps (if discretization is used)
        self._dim = dim
        
        if(self._dim == Multilevel_Train_Dimension.five ):
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
        else:
            # 1d for testing
            # Training set definition
            self.s_0_trainInterval = [118.0, 120.0]
            self.sigma_trainInterval = [0.2, 0.2]
            self.mu_trainInterval = [0.05, 0.05]
            self.T_trainInterval = [10.0, 10.0]
            self.K_trainInterval = [410.0, 410.0]
            
            # Test set modification: (reducing test interval slightly for better testing)
            self.s_0_h = 0.0
            self.sigma_h = 0.0
            self.mu_h = 0.0
            self.T_h = 0.0              
            self.K_h = 0.0
        
    def trainingSet(self, m, trainSeed=None, approx=False):  
        #np.random.seed(trainSeed) 
        # 1. Draw parameter samples for training
        s_0 = (self.s_0_trainInterval[1] - self.s_0_trainInterval[0]) * np.random.random_sample(m) + self.s_0_trainInterval[0]
        sigma = (self.sigma_trainInterval[1] - self.sigma_trainInterval[0]) * np.random.random_sample(m) + self.sigma_trainInterval[0]
        mu = (self.mu_trainInterval[1] - self.mu_trainInterval[0]) * np.random.random_sample(m) + self.mu_trainInterval[0]
        T = (self.T_trainInterval[1] - self.T_trainInterval[0]) * np.random.random_sample(m) + self.T_trainInterval[0]
        K = (self.K_trainInterval[1] - self.K_trainInterval[0]) * np.random.random_sample(m) + self.K_trainInterval[0]

        if (self._opt == Multilevel_Train_Case.Milstein):
 
            # 2. Compute paths
            
            h = T[:]/ self._steps
            s = s_0
            # loop through the array for 10 steps
            for i in range(self._steps):
                # do something with the array
                z=np.random.normal(0.0, 1.0, m)
                s= s[:] + mu[:] *s[:] * h[:] +sigma[:] * s[:] *np.sqrt(h[:])*z[:] + 0.5 *sigma[:] *s_0[:] *sigma[:] * ((np.sqrt(h[:])*z[:])**2-h[:]) 
            # 3. Calculate and return payoffs
            payoffs=np.exp(-mu[:] * T[:])* np.maximum(s[:] - K[:], 0.)
            return np.stack((s_0,sigma,mu,T,K),axis=1), payoffs.reshape([-1,1]), None
        elif (self._opt == Multilevel_Train_Case.GBM_Path_Solution):
            #3. sets of random returns
            h=T[:]
            z=np.random.normal(0.0,1.0,m)
            #piecewise multiply of s= s_0[:] * np.exp((mu-sigma*sigma/2)*h+sigma*np.sqrt(h)*z[:])
            s= np.multiply(s_0[:],np.exp((mu[:] -0.5*sigma[:] *sigma[:] )*h[:] +sigma[:] *np.sqrt(h[:] )*z[:]))
            payoffs=np.exp(-mu[:]  * T[:] ) * np.maximum(s[:] - K[:] , 0.0)
            return np.stack((s_0,sigma,mu,T,K),axis=1), payoffs.reshape([-1,1]), None
        else:
            # B.S. formula
            d1 = (np.log(s_0[:]/K[:]) + (mu[:] + 0.5 * sigma[:] * sigma[:]) * T[:]) / (sigma[:] * np.sqrt(T[:]))
            d2 = d1[:] - sigma[:] * np.sqrt(T[:])
            price = s_0[:] * norm.cdf(d1[:]) - np.exp(-mu[:] *T[:] ) * K[:] * norm.cdf(d2[:])
            return np.stack((s_0,sigma,mu,T,K),axis=1), price.reshape([-1,1]), None
     
    def testSet(self, num, testSeed=None):
        #np.random.seed(testSeed)
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
        d1 = (np.log(s_0[:]/K[:]) + (mu[:] + 0.5 * sigma[:] * sigma[:]) * T[:]) / (sigma[:] * np.sqrt(T[:]))
        d2 = d1[:] - sigma[:] * np.sqrt(T[:])
        price = s_0[:] * norm.cdf(d1[:]) - np.exp(-mu[:] * T[:]) * K[:] * norm.cdf(d2[:])
        return np.stack((s_0,sigma,mu,T,K),axis=1), price.reshape([-1,1]), None, None
    
## Code in loop
# # Convert the function and condition to NumPy functions
# np_func = lambda i, s: my_Milstein(i, self._steps, s, mu, sigma, T, K)[1]  # keep only the second output of my_func
# np_cond = lambda i, s: my_Cond(i, s)
# # Create the initial input
# i = 0
# s = s_0
# # Run the while loop using NumPy
# s = np.while_loop(np_cond, np_func, [i, s])

## Helper functions
# # Define the function to be iterated
# def my_Milstein(i, steps, s, mu, sigma, T, m):
#     h = T[:] / steps
#     z = np.random.normal(0.0, 1.0, m)
#     s = s[:] + mu[:] * s[:] * h + sigma[:] * s[:] * np.sqrt(h) * z[:] + 0.5 * sigma[:] * s[:] * sigma[:] * ((np.sqrt(h) * z[:]) ** 2 - h)
#     return i + 1, s

# # Define the loop condition
# def my_Cond(i, s):
#     return i < s

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