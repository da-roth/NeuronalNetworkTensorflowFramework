import time
import tensorflow as tf2
#print("TF version =", tf2.__version__)
# we want TF 2.x
assert tf2.__version__ >= "2.0"
# disable eager execution etc
tf = tf2.compat.v1
tf.disable_eager_execution()

try:
    from TrainingOptionEnums import *
    from Neural_Approximator import *
    from normalize_data import *
    from TrainingSettings import *
except ModuleNotFoundError:
    #print("")
    from montecarlolearning.TrainingOptionEnums import *
    from montecarlolearning.Neural_Approximator import *
    from montecarlolearning.normalize_data import *
    from montecarlolearning.TrainingSettings import *

def train_and_test(generator,  
         regressor,
         TrainingSettings):
    
    if generator.TrainMethod == TrainingMethod.Standard:
        # 1. Simulation of training set, but only for max(sizes), other sizes will use these
        #print("simulating training, valid and test sets")
        xTrain, yTrain, _unused = generator.trainingSet(max(generator.trainingSetSizes), trainSeed=generator.dataSeed)
        xTest, yTest, _unused, _unused2 = generator.testSet(num=generator.nTest, testSeed=generator.testSeed)
        #print("done")

        # 2. Neural network initialization 
        #print("initializing neural appropximator")
        #print("done")

        regressor.initializeData(xTrain, yTrain)
        
        predvalues = {}    
        preddeltas = {}
        ## 3. Loop over train set sizes
        for size in generator.trainingSetSizes:         
            print("\nsize %d" % size)
            
            ###
            ### Vanilla net:
            ###
            
            # Prepare: normalize dataset and initialize tf graph
            regressor.prepare(size)
            
            # 4. Train network
            t0 = time.time()
            regressor.train("standard training",TrainingSettings.epochs,TrainingSettings.learningRateSchedule,TrainingSettings.batchesPerEpoch,TrainingSettings.minBatchSize,xTest=xTest,yTest=yTest)      
            
            # 5. Predictions on test data
            predictions = regressor.predict_values(xTest)
            predvalues[("standard", size)] = predictions
            t1 = time.time()
        return xTest, yTest, predvalues
         
    elif generator.TrainMethod == TrainingMethod.GenerateDataDuringTraining:
        # Parameters for GenerateDataDuringTraining
        TrainingSettings.minBatchSize = 1
        
        # 1. Simulation of initial training set
        #print("Simulating initial training set and test set")
        initial_sample_amount = max(generator.trainingSetSizes[0],100000) # to get a proper batch normalization
        xTrain, yTrain, _unused = generator.trainingSet(initial_sample_amount, trainSeed=generator.dataSeed)
        xTest, yTest, _unused, _unused2 = generator.testSet(num=generator.nTest, testSeed=generator.testSeed)
        #print("done")
        
        # 2. Neural network initialization 
        #print("initializing neural appropximator")
        regressor = Neural_Approximator(xTrain, yTrain)
        # Prepare: normalize dataset and initialize tf graph
        regressor.prepare(initial_sample_amount, False, hiddenNeurons, hiddenLayers, activationFunctionsHidden, activationFunctionOutput, weight_seed=weightSeed)
        #print("done")        
        
        # 3. First training step
        regressor.train("standard training",TrainingSettings.epochs,TrainingSettings.learningRateSchedule,TrainingSettings.batchesPerEpoch,TrainingSettings.minBatchSize)
        
        predvalues = {}    
        preddeltas = {}
        # 4. Train loop over remaining training steps
        for i in range(1,sizes[1]):
            #print('Training step ' + str(i) + ' will be done')
            xTrain, yTrain, _unused = generator.trainingSet(sizes[0], trainSeed=i)
            #regressor.storeNewDataAndNormalize(xTrain,  yTrain, _unused, sizes[0])
            
            # 4. Train network
            t0 = time.time()
            regressor.train("standard training",TrainingSettings.epochs,TrainingSettings.learningRateSchedule,TrainingSettings.batchesPerEpoch,TrainingSettings.minBatchSize, reinit = False)
            t1 = time.time()
            
            if i % testFrequency == 0:
                predictions = regressor.predict_values(xTest)
                errors = predictions - yTest
                rmse = np.sqrt((errors ** 2).mean(axis=0))
                print('RMSE after ' + str(i) + ' training steps is ' + str(rmse) )
                
        
        # 4. Predictions on test data
        predictions = regressor.predict_values(xTest)
        predvalues[("standard", generator.nTest)] = predictions
        return xTest, yTest, predvalues
        
    else:
       print('Training method not recognized')
                    
            
            
            

