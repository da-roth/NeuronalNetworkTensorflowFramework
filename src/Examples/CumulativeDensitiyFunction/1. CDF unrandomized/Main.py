### Copy of Main.py as a juypter notebook to visualize results

###
### 0. Import packages and references
###
### Import framework
import os
file_path = os.path.abspath(os.path.join(os.getcwd() , '..', 'NNFramework', 'packages.py'))
exec(open(file_path).read())
### - dataSeed = seed for simulations or (for csv input) for randomization of csv
dataSeed = 1 
weightSeed = 1 

###
### 1. Training data
###

generator = CDF()
sizes = [10000,100] # [sizePerTrainingStep, trainingSteps]
nTest = 2000 # Test set size


###
### 2. Set Nueral network structure / Hyperparameters
### 

hiddenNeurons = 20              # we use equal neurons for each hidden layer
hiddenLayers = 2                 # amount of hidden layers
activationFunctions = tf.nn.tanh    # activation functions of hidden layers
###
### 3. Train network
###
trainingMethod = TrainingMethod.GenerateDataDuringTraining

xTest, yTest, yPredicted = train_and_test(generator, sizes, nTest, dataSeed, None, weightSeed, hiddenNeurons, hiddenLayers, activationFunctions, trainingMethod)
    
###
### 3. Study results
###   

# show predicitions
plot_results("CDF random inputs", yPredicted, xTest, "x", "CDF(x)", yTest, sizes, True, False, None, trainingMethod)