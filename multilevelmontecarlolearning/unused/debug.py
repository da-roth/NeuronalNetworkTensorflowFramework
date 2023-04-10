#%reset -f

#Multilevel algorithm using 8 networks
#For more detailed explanations of the training and model parameters
#see Gerstner et al. "Multilevel Monte Carlo learning." arXiv preprint arXiv:2102.08734 (2021).

#Basic network framework according to Beck, Christian, et al. "Solving the Kolmogorov PDE by means of deep learning." Journal of Scientific Computing 88.3 (2021): 1-28.
#The framework was modified in such a way that it generates networks for each of the level estimators

import os
mainDirectory = os.path.abspath(os.path.join(os.getcwd() ))
packageFile = os.path.abspath(os.path.join(mainDirectory, 'montecarlolearning', 'packages.py'))
exec(open(packageFile).read())
 

Generator = GBM_Multilevel()
  
Regressor = Neural_Approximator_Multilevel()
Regressor.set_hiddenNeurons(50)
Regressor.set_hiddenLayers(2)

TrainSettings = TrainingSettings()
TrainSettings.set_learning_rate_schedule([0.01, 0.1, 1000])
TrainSettings.set_test_frequency(150)
TrainSettings.set_mcRounds(100)
TrainSettings.set_nTest(200)
TrainSettings.set_samplesPerStep([75000, 1817, 690, 264, 93, 33, 12, 5])
TrainSettings.set_trainingSteps([150000,150,150,150,1500,150,150,150])

train_and_test_Multilevel(Generator, Regressor, TrainSettings)