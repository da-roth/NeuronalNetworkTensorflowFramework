### Copy of Main.py as a juypter notebook to visualize results

###
### 0. Import packages and references
###
### Import framework
import os
mainDirectory = os.getcwd()
packageFile = os.path.abspath(os.path.join(mainDirectory, 'montecarlolearning', 'packages.py'))
exec(open(packageFile).read())

###
### 1. Training data
###
#from CDF import *
###
### 1. Training data
###
#from CDF import *
Generator = DataImporter()
# 10^4 data data points
Generator.set_path(os.path.join(mainDirectory, 'src', 'Examples', 'CumulativeDensitiyFunction', 'cdf_deterministic_data.csv'))
Generator.set_inputName('x')
Generator.set_outputName('CDF(x)')
Generator.set_trainTestRatio(0.8)
Generator.set_randomized(False)
Generator.set_trainingSetSizes([100,1000,10000])
Generator.set_dataSeed(1)

###
### 2. Set Nueral network structure / Hyperparameters
### 

Regressor = Neural_Approximator()
Regressor.set_Generator(Generator)
Regressor.set_hiddenNeurons(20)
Regressor.set_hiddenLayers(3)
Regressor.set_activationFunctionsHidden([tf.nn.tanh])
Regressor.set_weight_seed(1)

###
### 3. Training settings
### 

TrainSettings = TrainingSettings()
TrainSettings.set_epochs(20)

###
### 4. Train and evaluate
###
xTest, yTest, yPredicted = train_and_test(Generator, Regressor, TrainSettings)
plot_results("CDF unrandomized deterministic inputs", yPredicted, xTest, yTest, Generator)