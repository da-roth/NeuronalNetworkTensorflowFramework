import numpy as np
from scipy.stats import norm


__version__ = 'dev'

def hello_world():
    print("This is my first pip package!")
 
def hello_world2():
    print("This is my first pip package!")   

# helper analytics    
def bsPrice(spot, strike, vol, T):
    d1 = (np.log(spot/strike) + 0.5 * vol * vol * T) / vol / np.sqrt(T)
    d2 = d1 - vol * np.sqrt(T)
    return spot * norm.cdf(d1) - strike * norm.cdf(d2)    