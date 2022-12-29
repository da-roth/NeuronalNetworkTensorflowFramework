# Monte Carlo Learning

- The first goal of this repo is to provide a flexible neural network training framework, not only able to handle the usual case of fixed inputs (through e.g. .csv files), but to allow data generation (on the fly during the training process) through self-written functions. 

- The second goal is to give an introduction to neural network regression from the Monte Carlo point of view. The first section of the Monte Carlo learning documentation: https://github.com/da-roth/NeuronalNetworkTensorflowFramework/blob/main/documentation.pdf
will investigate the following three topics:
    1. Random selection and random inputs as training data
    2. Curse of dimensionality within the loss function approximation
    3. Approximation of outputs as training data
Therefore, we implemented examples from scratch, for which little prior knowledge is necessary.

# Multilevel Monte Carlo learning

The idea to study the different data generation approaches stems from
https://arxiv.org/abs/2102.08734v1

The basis of the code base which will be modified continuously stems from
https://github.com/differential-machine-learning/notebooks

Tutorial used for package creation:
https://github.com/MichaelKim0407/tutorial-pip-package