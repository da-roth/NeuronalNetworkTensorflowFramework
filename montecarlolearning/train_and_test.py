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

def train_and_test(Generator,  
         Regressor,
         TrainingSettings):
    
    if Generator.TrainMethod == TrainingMethod.Standard:
        # 1. Simulation of training set, but only for max(sizes), other sizes will use these
        #print("simulating training, valid and test sets")
        xTrain, yTrain, _unused = Generator.trainingSet(max(Generator.trainingSetSizes), trainSeed=Generator.dataSeed)
        xTest, yTest, _unused, _unused2 = Generator.testSet(num=TrainingSettings.nTest, testSeed=Generator.testSeed)
        #print("done")

        # 2. Neural network initialization 
        #print("initializing neural appropximator")
        #print("done")

        Regressor.initializeData(xTrain, yTrain)
        
        predvalues = {}    
        preddeltas = {}
        ## 3. Loop over train set sizes
        for size in Generator.trainingSetSizes:         
            print("\nsize %d" % size)
            
            ###
            ### Vanilla net:
            ###
            
            # Prepare: normalize dataset and initialize tf graph
            Regressor.prepare(size)
            
            # 4. Train network
            t0 = time.time()
            Regressor.train("standard training",TrainingSettings,xTest=xTest,yTest=yTest)      
            
            # 5. Predictions on test data
            predictions = Regressor.predict_values(xTest)
            predvalues[("standard", size)] = predictions
            t1 = time.time()
        return xTest, yTest, predvalues
         
    elif Generator.TrainMethod == TrainingMethod.GenerateDataDuringTraining:
        # Parameters for GenerateDataDuringTraining

        # 1. Simulation of initial training set
        #print("Simulating initial training set and test set")
        initial_sample_amount = max(TrainingSettings.SamplesPerStep,10000) # to get a proper batch normalization
        xTrain, yTrain, _unused = Generator.trainingSet(initial_sample_amount, trainSeed=Generator.dataSeed)
        xTest, yTest, _unused, _unused2 = Generator.testSet(num=TrainingSettings.nTest, testSeed=Generator.testSeed)
        #print("done")
        
        # 2. Neural network initialization 
        #print("initializing neural appropximator")
        Regressor.initializeData(xTrain, yTrain)
        # Prepare: normalize dataset and initialize tf graph
        Regressor.prepare(initial_sample_amount)
        #print("done")        
        
        # 3. First training step
        Regressor.train("standard training",TrainingSettings)
        
        predvalues = {}    
        preddeltas = {}
        # 4. Train loop over remaining training steps
        for i in range(1,TrainingSettings.TrainingSteps):
            #print('Training step ' + str(i) + ' will be done')
            xTrain, yTrain, _unused = Generator.trainingSet(TrainingSettings.SamplesPerStep, trainSeed=i)
            #Regressor.storeNewDataAndNormalize(xTrain,  yTrain, _unused, sizes[0])
            
            # 4. Train network
            t0 = time.time()
            Regressor.train("standard training",TrainingSettings, reinit = False)
            t1 = time.time()
            
            if i % TrainingSettings.testFrequency == 0:
                predictions = Regressor.predict_values(xTest)
                errors = predictions - yTest
                rmse = np.sqrt((errors ** 2).mean(axis=0))
                print('RMSE after ' + str(i) + ' training steps is ' + str(rmse) )
                
        
        # 4. Predictions on test data
        predictions = Regressor.predict_values(xTest)
        predvalues[("standard", TrainingSettings.nTest)] = predictions
        return xTest, yTest, predvalues
        
    else:
       print('Training method not recognized')
