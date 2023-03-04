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
#sizes = [1000,100] # [sizePerTrainingStep, trainingSteps]
#nTest = 2000 # Test set size

Generator = CDF()
#print(Generator.path == None)
Generator.set_inputName('x')
Generator.set_outputName('CDF(x)')
Generator.set_trainingSetSizes([1000,100])
Generator.set_nTest(2000)

###
### 2. Set Nueral network structure / Hyperparameters
### 

Regressor = Neural_Approximator()
Regressor.set_Generator(Generator)
Regressor.set_hiddenNeurons(20)
Regressor.set_hiddenLayers(3)
Regressor.set_activationFunctionsHidden([tf.nn.tanh])

#hiddenNeurons = 20               # we use equal neurons for each hidden layer
#hiddenLayers = 3                # amount of hidden layers
#activationFunctionsHidden = [tf.nn.tanh]   # activation functions of hidden layers

TrainSettings = TrainingSettings()
TrainSettings.set_epochs(20)
TrainSettings.set_min_batch_size(1)

###
### 3. Train network and Study results
### Comment: For different trainingSetSizes the neural network reset and not saved, hence train and evaluation of yPredicted are done together currently
###
xTest, yTest, yPredicted = train_and_test(Generator, Regressor, TrainSettings)
plot_results("CDF unrandomized deterministic inputs", yPredicted, xTest, yTest, Generator)

# ###
# ### 2. Set Nueral network structure / Hyperparameters
# ### 

# hiddenNeurons = 20                      # we use equal neurons for each hidden layer
# hiddenLayers = 3                        # amount of hidden layers
# activationFunctionsHidden = tf.nn.tanh   # activation functions of hidden layers
# batches_per_epoch = sizes[0]

# ###
# ### 3. Train network
# ###

# trainingMethod = TrainingMethod.GenerateDataDuringTraining
# xTest, yTest, yPredicted = train_and_test(generator, sizes, nTest, dataSeed, None, weightSeed, hiddenNeurons, hiddenLayers, activationFunctionsHidden, trainingMethod = trainingMethod, batches_per_epoch = batches_per_epoch)
    
# ###
# ### 3. Study results
# ###   

# # show predicitions
# plot_results("CDF random inputs", yPredicted, xTest, "x", "CDF(x)", yTest, sizes, True, False, None, trainingMethod)