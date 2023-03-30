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
         TrainSettings):
    
    if Generator.TrainMethod == TrainingMethod.Standard:
        # 1. Simulation of training set, but only for max(sizes), other sizes will use these
        #print("simulating training, valid and test sets")
        xTrain, yTrain, _unused = Generator.trainingSet(max(Generator.trainingSetSizes), trainSeed=Generator.dataSeed)
        xTest, yTest, _unused, _unused2 = Generator.testSet(num=TrainSettings.nTest, testSeed=Generator.testSeed)
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
            Regressor.train("standard training",TrainSettings,xTest=xTest,yTest=yTest)      
            
            # 5. Predictions on test data
            predictions = Regressor.predict_values(xTest)
            predvalues[("standard", size)] = predictions
            t1 = time.time()
        return xTest, yTest, predvalues
         
    elif Generator.TrainMethod == TrainingMethod.GenerateDataDuringTraining:
        # Parameters for GenerateDataDuringTraining

        # 1. Simulation of initial training set
        #print("Simulating initial training set and test set")
        initial_sample_amount = max(TrainSettings.SamplesPerStep,10000) # to get a proper batch normalization
        
        Regressor.initializeAndResetGraph()
        with Regressor.graph.as_default():  
            xTrain, yTrain, _unused = Generator.trainingSet(initial_sample_amount, trainSeed=Generator.dataSeed)
            xTest, yTest, _unused, _unused2 = Generator.testSet(num=TrainSettings.nTest, testSeed=Generator.testSeed)
            #print("done")
            
            # 2. Neural network initialization 
            #print("initializing neural appropximator")
            Regressor.initializeData(xTrain, yTrain)
            # Prepare: normalize dataset and initialize tf graph
            Regressor.prepare(initial_sample_amount, TrainSettings.nTest)
            #print("done")        
          
            # 3. First training step
            Regressor.train("standard training",TrainSettings)
            
            predvalues = {}    
            preddeltas = {}
            # 4. Train loop over remaining training steps
            file_out = open('output.csv', 'w')
            file_out.write('train_steps, RMSE, Max_Error \n ')
            for i in range(1,TrainSettings.TrainingSteps):
                #print('Training step ' + str(i) + ' will be done')
                xTrain, yTrain, _unused = Generator.trainingSet(TrainSettings.SamplesPerStep, trainSeed=i)
                
                # ToDo: rethink this. It doesn't work without this, see e.g. closed path gbm. 
                # Since data is generated each time, the first normalization is not correct later...
                # Idea: Perhaps with max/min of intervals, to overcome border cases...?
                Regressor.storeNewDataAndNormalize(xTrain,  yTrain, _unused, TrainSettings.SamplesPerStep)
                
                # 4. Train network
                t0 = time.time()
                Regressor.train("standard training",TrainSettings, reinit = False)
                t1 = time.time()
                
                # Generate output file and print results during training

                if (i+1) % TrainSettings.testFrequency == 0:
                    isTraining = False
                    predictions = Regressor.predict_values(xTest, isTraining)
                    errors = predictions - yTest
                    if isinstance(errors, tf.Tensor):
                        errors = errors.eval(session=Regressor.session)
                    L_2 = np.sqrt((errors ** 2).mean(axis=0))
                    L_infinity = np.max(np.abs(errors))
                    print('RMSE after ' + str(i+1) + ' training steps is ' + str(L_2) )
                    file_out.write('%i, %f, %f \n' % (i+1, L_2,L_infinity)) 
                    file_out.flush()
                    
            
            # 4. Predictions on test data
            isTraining = False
            predictions = Regressor.predict_values(xTest, isTraining)
            predvalues[("standard", TrainSettings.nTest)] = predictions
            # Last entry and print of error:
            errors = predictions - yTest
            if isinstance(errors, tf.Tensor):
                errors = errors.eval(session=Regressor.session)
            L_2 = np.sqrt((errors ** 2).mean(axis=0))
            L_infinity = np.abs(errors).max(axis=0)
            print('RMSE after training is ' + str(L_2) )
            print('max error  after training is ' + str(L_infinity) )
            # file_out.write('%i, %f, %f \n' % (i+1, L_2,L_infinity)) 
            # file_out.flush()
                    
            return xTest, yTest, predvalues
        
    else:
       print('Training method not recognized')
