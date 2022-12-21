import time

try:
    from TrainingMethod import *
    from Neural_Approximator import *
    from normalize_data import *
except ModuleNotFoundError:
    #print("")
    from montecarlolearning.TrainingMethod import *
    from montecarlolearning.Neural_Approximator import *
    from montecarlolearning.normalize_data import *

def train_and_test_with_differentials(generator, 
         sizes, 
         nTest, 
         dataSeed=None, 
         testSeed=None, 
         weightSeed=None, 
         deltidx=0):

    # 1. Simulation of training set, but only for max(sizes), other sizes will use these
    print("simulating training, valid and test sets")
    xTrain, yTrain, dydxTrain = generator.trainingSet(max(sizes), trainSeed=dataSeed)
    xTest, yTest, dydxTest, vegas = generator.testSet(num=nTest, testSeed=testSeed)
    print("done")

    # 2. Neural network initialization 
    print("initializing neural appropximator")
    regressor = Neural_Approximator(xTrain, yTrain, dydxTrain)
    print("done")
    
    predvalues = {}    
    preddeltas = {}
    ## Loop over train set sizes
    for size in sizes:         
        print("\nsize %d" % size)
        
        ###
        ### Vanilla net:
        ###
        
        # Prepare: normalize dataset and initialize tf graph
        regressor.prepare(size, False, weight_seed=weightSeed)
         
        # Train network
        t0 = time.time()
        regressor.train("standard training")
        
        # Predictions on test data
        predictions, deltas = regressor.predict_values_and_derivs(xTest)
        predvalues[("standard", size)] = predictions
        preddeltas[("standard", size)] = deltas[:, deltidx]
        t1 = time.time()
        
        ###
        ### Differential net:
        ###
        
        # Prepare: normalize dataset and initialize tf graph
        regressor.prepare(size, True, weight_seed=weightSeed)
          
        # Train network  
        t0 = time.time()
        regressor.train("differential training")
        
        # Predictions on test data
        predictions, deltas = regressor.predict_values_and_derivs(xTest)
        predvalues[("differential", size)] = predictions
        preddeltas[("differential", size)] = deltas[:, deltidx]
        t1 = time.time()
        
    return xTest, yTest, dydxTest[:, deltidx], vegas, predvalues, preddeltas

