import time

from Neural_Approximator import *
from TrainingMethod import *

def train_and_test(generator, 
         sizes, 
         nTest, 
         dataSeed=None, 
         testSeed=None, 
         weightSeed=None, 
         hiddenNeurons=20,
         hiddenLayers=2,
         activationFunctions=tf.nn.softmax,
         trainingMethod = TrainingMethod.Standard,
         testFrequency = 10,
         outputDimension = 1,
         deltidx=0,
         epochs=1, 
         learning_rate_schedule=[
            (0.0, 0.01), 
            (0.2, 0.001), 
            (0.4, 0.0001), 
            (0.6, 0.00001), 
            (0.8, 0.000001)], 
         batches_per_epoch=10,
         min_batch_size=20):
    
    if trainingMethod == TrainingMethod.Standard:
        # 1. Simulation of training set, but only for max(sizes), other sizes will use these
        print("simulating training, valid and test sets")
        xTrain, yTrain, _unused = generator.trainingSet(max(sizes), trainSeed=dataSeed)
        xTest, yTest, _unused, _unused2 = generator.testSet(num=nTest, testSeed=testSeed)
        print("done")

        # 2. Neural network initialization 
        print("initializing neural appropximator")
        regressor = Neural_Approximator(xTrain, yTrain)
        print("done")
        
        predvalues = {}    
        preddeltas = {}
        ## 3. Loop over train set sizes
        for size in sizes:         
            print("\nsize %d" % size)
            
            ###
            ### Vanilla net:
            ###
            
            # Prepare: normalize dataset and initialize tf graph
            regressor.prepare(size, False, hiddenNeurons, hiddenLayers, activationFunctions, weight_seed=weightSeed)
            
            # 4. Train network
            t0 = time.time()
            regressor.train("standard training",epochs,learning_rate_schedule,batches_per_epoch,min_batch_size,xTest=xTest,yTest=yTest)      
            
            # 5. Predictions on test data
            predictions = regressor.predict_values(xTest)
            predvalues[("standard", size)] = predictions
            t1 = time.time()
        return xTest, yTest, predvalues
         
    elif trainingMethod == TrainingMethod.GenerateDataDuringTraining:
        # 1. Simulation of initial training set
        #print("Simulating initial training set and test set")
        xTrain, yTrain, _unused = generator.trainingSet(sizes[0], trainSeed=0)
        xTest, yTest, _unused, _unused2 = generator.testSet(num=nTest, testSeed=testSeed)
        #print("done")
        
        # 2. Neural network initialization 
        #print("initializing neural appropximator")
        regressor = Neural_Approximator(xTrain, yTrain)
        # Prepare: normalize dataset and initialize tf graph
        regressor.prepare(sizes[0], False, hiddenNeurons, hiddenLayers, activationFunctions, weight_seed=weightSeed)
        #print("done")        
        
        # 3. First training step
        regressor.train("standard training",epochs,learning_rate_schedule,batches_per_epoch,min_batch_size)
        
        predvalues = {}    
        preddeltas = {}
        # 4. Train loop over remaining training steps
        for i in range(1,sizes[1]):
            #print('Training step ' + str(i) + ' will be done')
            xTrain, yTrain, _unused = generator.trainingSet(sizes[0], trainSeed=i)
            regressor.storeNewDataAndNormalize(xTrain,  yTrain, _unused, sizes[0])
            
            # 4. Train network
            t0 = time.time()
            regressor.train("standard training",epochs,learning_rate_schedule,batches_per_epoch,min_batch_size)
            t1 = time.time()
            
            if i % testFrequency == 0:
                predictions = regressor.predict_values(xTest)
                errors = predictions - yTest
                rmse = np.sqrt((errors ** 2).mean(axis=0))
                print('RMSE after ' + str(i) + ' training steps is ' + str(rmse) )
                
        
        # 4. Predictions on test data
        predictions = regressor.predict_values(xTest)
        predvalues[("standard", nTest)] = predictions
        return xTest, yTest, predvalues
        
    else:
       print('Training method not recognized')
                    
            
            
            

