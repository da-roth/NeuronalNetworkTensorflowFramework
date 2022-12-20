# MonteCarloLearning

- The first goal of this repo is to provide a flexible neural network training framework, not only able to handle the usual case of fixed inputs (e.g. .csv files), but to allow data generation (on the fly during the training process) through self-written functions. 

- The second goal is to give an introduction to the idea of understanding training data selection from the Monte Carlo point of view by showing examples from scratch, for which little prior knowledge is necessary. Hence, the first introductional example is the training of the cumulative normal densitity function, which is explained in-depth in the documentation.pdf and in jupyter notebooks to recompute in /Examples/CumulativeDensityFunction.

The idea to study the different data generation approaches stems from
https://arxiv.org/abs/2102.08734v1

The basis of the code base which will be modified continuously stems from
https://github.com/differential-machine-learning/notebooks

While in this work I define Monte Carlo learning as the approach of using random sampled inputs and only approximations of the output, I believe that studying this approach may help to understand neural network training in general.

For high-dimensional problems, trying to first generate training data and store them in files will sooner or later lead to infeasible file sizes. Under these circumstances, it is necessary to be able to compute training data on the fly during the neural network training process.


Tutorial used for package creation:
https://github.com/MichaelKim0407/tutorial-pip-package