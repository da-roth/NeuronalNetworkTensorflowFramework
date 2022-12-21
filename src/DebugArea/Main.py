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

x = ['active','mean_dist_stanlag','dist_capital_stanlag','ch_dist_capital_stanlag','territorial','government','osk_a_stanlag','osk_b_stanlag','closed_aut','elect_aut','elect_dem','lib_dem','ln_brd_lag','low_intlev','high_intlev','internationalised','infmorMA_nn','frac_std','y_active_std']
y = ['ordinal_theta']
sizes = [544] # training set sizes. Performed one after the other and might be compared
nTest = 0.9 # Test set is given through a ratio of 0.8 in generator
generator = DataImporter('C:\dev\MonteCarloLearning\src\socialScience\second_NN_new.csv',x,y,nTest) 

###
### 2. Set Nueral network structure / Hyperparameters
### 


hiddenNeurons = 20               # we use equal neurons for each hidden layer
hiddenLayers = 50              # amount of hidden layers
activationFunctionsHidden = tf.nn.tanh   # activation functions of hidden layers
activationFunctionOutput = tf.nn.relu

epochs=100
learning_rate_schedule=[
    (0.0, 0.05), 
    (0.2, 0.025), 
    (0.4, 0.01), 
    (0.6, 0.001), 
    (0.8, 0.0001)] 
batches_per_epoch=150
min_batch_size=150

outputDimension = 1,
biasNeuron = True

###
### 3. Train network
###
trainingMethod = TrainingMethod.Standard
xTest, yTest, yPredicted = train_and_test(generator, sizes, nTest, dataSeed, None, weightSeed, hiddenNeurons, hiddenLayers, activationFunctionsHidden, trainingMethod = trainingMethod, epochs = epochs,learning_rate_schedule=learning_rate_schedule,batches_per_epoch=batches_per_epoch,outputDimension=outputDimension,biasNeuron=biasNeuron,min_batch_size=min_batch_size,activationFunctionOutput=activationFunctionOutput)
    
### 4. Study results
###   

plot_results("Results", yPredicted, xTest, "x", "y", yTest, sizes, True, False, None, trainingMethod)