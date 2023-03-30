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
Generator = Multilevel_GBM(Multilevel_Train_Case.Milstein, 0, Multilevel_Train_Dimension.one)
Generator.set_inputName('S')
Generator.set_outputName('P_0')

GeneratorLevel1 = Multilevel_GBM(Multilevel_Train_Case.LevelEstimator, 0, Multilevel_Train_Dimension.one)
GeneratorLevel1 .set_inputName('x')
GeneratorLevel1 .set_outputName('P_1-P_0')

GeneratorLevel2 = Multilevel_GBM(Multilevel_Train_Case.LevelEstimator, 1, Multilevel_Train_Dimension.one)
GeneratorLevel2 .set_inputName('x')
GeneratorLevel2 .set_outputName('P_2-P_1')

GeneratorLevel3 = Multilevel_GBM(Multilevel_Train_Case.LevelEstimator, 2, Multilevel_Train_Dimension.one)
GeneratorLevel3 .set_inputName('x')
GeneratorLevel3 .set_outputName('P_3-P_2')

GeneratorLevel4 = Multilevel_GBM(Multilevel_Train_Case.LevelEstimator, 3, Multilevel_Train_Dimension.one)
GeneratorLevel4 .set_inputName('x')
GeneratorLevel4 .set_outputName('P_4-P_3')


###
### 2. Set Nueral network structure / Hyperparameters
### 

Regressor = Neural_Approximator()
Regressor.set_Generator(Generator)
Regressor.set_hiddenNeurons(20)
Regressor.set_hiddenLayers(2)
Regressor.set_activationFunctionsHidden(tf.nn.sigmoid)
Regressor.set_activationFunctionOutput(tf.nn.sigmoid)
Regressor.set_weight_seed(1)

TrainSettings = TrainingSettings()
TrainSettings.set_learning_rate_schedule=([(0.0, 0.0001),  (0.2, 0.0001),  (0.4, 0.0001), (0.6, 0.0001),  (0.8, 0.0001)] )
TrainSettings.set_min_batch_size(1)
TrainSettings.set_test_frequency(10)
TrainSettings.set_nTest(100000)
TrainSettings.set_samplesPerStep(20000)
TrainSettings.set_trainingSteps(20)

###
### 3. Train network and Study results
### Comment: For different trainingSetSizes the neural network reset and not saved, hence train and evaluation of yPredicted are done together currently
###

xTest, yTest, yPredicted = train_and_test(Generator, Regressor, TrainSettings)
TrainSettings.set_samplesPerStep(100000)
xTest2, yTest2, yPredictedLevel1 = train_and_test(GeneratorLevel1, Regressor, TrainSettings)