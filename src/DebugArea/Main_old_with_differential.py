### Main implementation for debugging purposes. For visualization and result interpretation use Main.ipynb.

###
### 0. Import packages and references
###

### Import framework
import os
mainDirectory = os.getcwd()
packageFile = os.path.abspath(os.path.join(mainDirectory,'src', 'NNFramework', 'packages.py'))
exec(open(packageFile).read())
### - dataSeed = seed for simulations or (for csv input) for randomization of csv
dataSeed = 1 
weightSeed = 1 

###
### 1. Set Nueral network structure / Training data / Hyperparameters
### 

# training set sizes. Performed one after the other and might be compared
sizes = [750, 7500]
# Test set size
nTest = 2000
# Neurons
hiddenNeurons = 40                  # we use equal neurons for each hidden layer
hiddenLayers = 2                    # amount of hidden layers
activationFunctions = tf.nn.softmax    # activation functions of hidden layers

###
### 2. Train network
###
differentialML = False

generator = BlackScholes()
#generator = DataImporter('C:/dev/MonteCarloLearning/src/BlackScholes/data/training1dimDataImpliedVolatility10E5.csv')

if not differentialML:
    ### Vanilla
    xTest, yTest, yPredicted = train_and_test(generator, sizes, nTest, dataSeed, None, weightSeed, hiddenNeurons, hiddenLayers, activationFunctions)
else: 
    ### Vanilla + Differential
    xTest, yTest, dydxTest, vegas, yPredicted, deltas = train_and_test_with_differentials(generator, sizes, nTest, dataSeed, None, weightSeed)
    
###
### 3. Study results
###   

# show predicitions
plot_results("Black & Scholes", yPredicted, xTest, "test inputs", "yPredicted", yTest, sizes, True, differentialML)

# show deltas
if differentialML:
    plot_results("Black & Scholes", deltas, xTest, "test inputs", "deltas", dydxTest, sizes, True, differentialML)
