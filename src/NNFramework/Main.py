### Main implementation for debugging purposes. For visualization and result interpretation use Main.ipynb.

###
### 0. Import packages and references
###
### - dataSeed = seed for simulations or (for csv input) for randomization of csv

exec(open('C:/dev/MonteCarloLearning/src/NNFramework/packages.py').read())
dataSeed = 1 
weightSeed = 1 


### 1. Training data
###
### - First option: use generation class that computes training data during training
### - Second option: use csv sheet

generator = BlackScholes()
#sizes = [750, 7500] # training set sizes. Performed one after the other and might be compared

#generator = DataImporter('C:/dev/MonteCarloLearning/src/BlackScholes/data/training1dimDataImpliedVolatility10E5.csv')
sizes = [1000,2000] # [sizePerTrainingStep, trainingSteps]
nTest = 2000 # Test set size


###
### 2. Set Nueral network structure / Hyperparameters
### 

hiddenNeurons = 40                  # we use equal neurons for each hidden layer
hiddenLayers = 2                    # amount of hidden layers
activationFunctions = tf.nn.softmax    # activation functions of hidden layers
###
### 3. Train network
###
trainingMethod = TrainingMethod.GenerateDataDuringTraining

xTest, yTest, yPredicted = train_and_test(generator, sizes, nTest, dataSeed, None, weightSeed, hiddenNeurons, hiddenLayers, activationFunctions, trainingMethod)
### 1. Training data
###
### - First option: use generation class that computes training data during training
### - Second option: use csv sheet

# 10^4 data data points
generator = DataImporter('C:/dev/MonteCarloLearning/src/Examples/2. CDF_MC/cdf_randomInputs_data.csv','x','CDF(x)',testDataPath='C:/dev/MonteCarloLearning/src/Examples/2. CDF_MC/cdf_randomInputs_data.csv') 
sizes = [100,1000,10000] # training set sizes. Performed one after the other and might be compared
nTest = None # Test set is given through a ratio of 0.8 in generator


###
### 2. Set Nueral network structure / Hyperparameters
### 

hiddenNeurons = 40               # we use equal neurons for each hidden layer
hiddenLayers = 4                 # amount of hidden layers
activationFunctions = tf.nn.sigmoid    # activation functions of hidden layers
###
### 3. Train network
###
trainingMethod = TrainingMethod.Standard

xTest, yTest, yPredicted = train_and_test(generator, sizes, nTest, dataSeed, None, weightSeed, hiddenNeurons, hiddenLayers, activationFunctions, trainingMethod)
    
###
### 3. Study results
###   

# show predicitions
plot_results("Black & Scholes", yPredicted, xTest, "test inputs", "yPredicted", yTest, sizes, True, False, None, trainingMethod)

### Todo: 

# For network:
# - individual neurons [40,30,20]
# - individual activation functions [tf.nn.softmax,tf.nn.relu]
# - individual output activation function

# For training:
# - epoch and batch size
# - learning rate, decay options
# - test frequency (currently only at the end)

# For general generators:
# - input selection: random, determinisitic
# - amount train steps
# - amount data per train step

# For DataImporter: 
# - define ration train/test
# - Set Input/Output dimension and columns of sheet

# For Plots:
# - generated on the fly: plots after x training steps and not only for the last of size (and first currently empty)
