### Copy of Main.py as a juypter notebook to visualize results

###
### 0. Import packages and references
###
### Import framework
import os
mainDirectory = os.getcwd()
packageFile = os.path.abspath(os.path.join(mainDirectory, 'montecarlolearning', 'packages.py'))
exec(open(packageFile).read())
### - dataSeed = seed for simulations or (for csv input) for randomization of csv
dataSeed = 1 
weightSeed = 1 

###
### 1. Training data
###
#from CDF import *
generator = GBM()
sizes = [1000,100] # [sizePerTrainingStep, trainingSteps]
nTest = 2000 # Test set size

###
### 2. Set Nueral network structure / Hyperparameters
### 

hiddenNeurons = 20                      # we use equal neurons for each hidden layer
hiddenLayers = 3                        # amount of hidden layers
activationFunctionsHidden = tf.nn.tanh   # activation functions of hidden layers

###
### 3. Train network
###

trainingMethod = TrainingMethod.GenerateDataDuringTraining
xTest, yTest, yPredicted = train_and_test(generator, sizes, nTest, dataSeed, None, weightSeed, hiddenNeurons, hiddenLayers, activationFunctionsHidden, trainingMethod = trainingMethod, batches_per_epoch = 1)
    
###
### 3. Study results
###   

# show predicitions
plot_results("CDF random inputs", yPredicted, xTest, "x", "CDF(x)", yTest, sizes, True, False, None, trainingMethod)