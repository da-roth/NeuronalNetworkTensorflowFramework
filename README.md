# Neural Network Regression Framework with Monte Carlo Learning

This framework is designed for regression-based neural network training tasks, with a focus on Monte Carlo learning. The aim is to provide a flexible and customizable environment for experimenting with neural networks and their application to various types of data input/generation methods. The repository contains a series of examples that demonstrate the core concepts, with increasing levels of complexity.

In addition to the provided examples, there is an [introduction PDF](https://github.com/da-roth/NeuronalNetworkTensorflowFramework/blob/main/documentation.pdf) that delves into the mathematical background of Monte Carlo learning.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Multilevel Monte Carlo Learning](#multilevel-monte-carlo-learning)
- [Contributing](#contributing)
- [License](#license)

## Overview

The documentation ([link](https://github.com/da-roth/NeuronalNetworkTensorflowFramework/blob/main/documentation.pdf)) guides you through the examples and presents the techniques and concepts in a clear and structured manner, starting from simple scenarios and gradually increasing in complexity. By studying the introduction and exploring the examples, you will gain a comprehensive understanding of Monte Carlo learning and its potential applications in neural network training tasks.

The project is divided into two main components:
1. Development of a flexible framework for regression-based neural network training tasks using two primary classes of data input/generation methods:
   - General file inputs for precomputed data usage
   - Individual functions for on-the-fly data generation during training
2. Investigation of neural networks' properties with respect to Monte Carlo learning techniques.

The repository also includes an extension that demonstrates the application of multilevel neural network training for the approximation of expected values of functional quantities on the solution of a stochastic differential equation (SDE), as described in the ["Multilevel Monte Carlo Learning"](https://arxiv.org/abs/2102.08734) article.


## Installation

There are two ways to set up the environment and install the required dependencies:

### 1. Manual Installation

To set up the environment and install the required dependencies, please follow these steps (see [here](https://github.com/da-roth/NeuronalNetworkTensorflowFramework/blob/main/local_environment_installation_guide.txt) for more detailed instructions):

1. Open command prompt and naviagte to folder (e.g. C\dev\NeuralNetworkTensorflowFramework) 
2. Create virtual environment
```
python -m venv .venvNN
```
3. Activate the virtual environment:
- For Windows:
  ```
  .venvNN\Scripts\activate.bat
  ```
- For Linux/Mac:
  ```
  source .venvNN/bin/activate
  ```
4. Install packages from requirements.txt, 
```
pip install -r requirements.txt
```
### 2. Package Installation
Alternatively, you can install the package through pip, which includes the required dependencies:
```
pip install git+https://github.com/da-roth/NeuronalNetworkTensorflowFramework#montecarlolearning
```

This installation method makes use of the provided `setup.py` file, allowing for a more streamlined installation process. This approach is particularly useful for running the example files in Google Colab, as it enables a seamless and efficient execution of the examples without the need for manual installation steps. By using this package installation method, you can quickly set up the environment and focus on exploring the examples and their outputs.


## Usage

After installing the required dependencies, you can run the example files in the `src/Examples` directory. The main framework is located in the `montecarlolearning` directory.

To use this framework for your own project, follow the steps below:

1. Specify the training data generator or importer. There are two options available:
    1. Use the **DataImporter** class provided in the framework to import data from files. For example:
      ```
      Generator = DataImporter()
      Generator.set_path(pathOfFile)
      ```
    2. Create your own data generator by inheriting from the **TrainingDataGenerator** abstract class provided in the framework. You can then override the **trainingSet** and **testSet** methods to generate your own custom training data. For example:
        ```
        class MyDataGenerator(TrainingDataGenerator):
          def trainingSet(self, m, trainSeed=None):
              # custom implementation for generating training data
              pass
          def testSet(self, num, testSeed=None):
              # custom implementation for generating test data
              pass
        Generator = MyDataGenerator()
        ```
    
2. Initialize the **Neural_Approximator** specify the neural network structure. Example: 
    ```
    Regressor = Neural_Approximator()
    Regressor.set_hiddenNeurons(20)
    Regressor.set_hiddenLayers(3)
    ```
3. Initialize the and specify **TrainingSettings**. Example:
    ```
    TrainSettings = TrainingSettings()
    TrainSettings.set_epochs(20)
    TrainSettings.set_batches_per_epoch(10)
    ```
4. Train and evaluate. Example:
    ```
    train_and_test(Generator, Regressor, TrainSettings)
    ```



## Examples

Several examples have been provided in the `src/Examples` directory, demonstrating the framework's capabilities and usage. A Google Colab executable link with all examples is also available and will be linked below.

Refer to the [documentation](https://github.com/da-roth/NeuronalNetworkTensorflowFramework/blob/main/src/documentation.pdf) for detailed explanations of the examples and the code.

### Cumulative Density Function Training Examples

- [Implementation](https://github.com/da-roth/NeuronalNetworkTensorflowFramework/blob/main/src/Examples/CumulativeDensitiyFunction/Implementation_CDF.ipynb)
- [Colab Implementation](https://github.com/da-roth/NeuronalNetworkTensorflowFramework/blob/main/src/Examples_Colab/CumulativeDensitiyFunction/Implementation_CDF.ipynb)

### Using Approximations As Training Data Examples

- One-dimensional geometric Brownian motion and the Black-Scholes closed solution for a European call option:
    - [Implementation](https://github.com/da-roth/NeuronalNetworkTensorflowFramework/blob/main/src/Examples/MonteCarloLearning/Implementation_GBM_ClosedSolution.ipynb)
    - [Colab Implementation](https://github.com/da-roth/NeuronalNetworkTensorflowFramework/blob/main/src/Examples_Colab/MonteCarloLearning/Implementation_GBM_ClosedSolution.ipynb)

- Now, use Monte Carlo simulation instead of the closed solution:
    - [Implementation](https://github.com/da-roth/NeuronalNetworkTensorflowFramework/blob/main/src/Examples/MonteCarloLearning/Implementation_GBM_MC.ipynb)
    - [Colab Implementation](https://github.com/da-roth/NeuronalNetworkTensorflowFramework/blob/main/src/Examples_Colab/MonteCarloLearning/Implementation_GBM_MC.ipynb)

- Five-dimensional training interval:
    - [Implementation](https://github.com/da-roth/NeuronalNetworkTensorflowFramework/blob/main/src/Examples/MonteCarloLearning/Implementation_GBM_5d.ipynb)
    - [Colab Implementation](https://github.com/da-roth/NeuronalNetworkTensorflowFramework/blob/main/src/Examples_Colab/MonteCarloLearning/Implementation_GBM_5d.ipynb)
  
- Multilevel Monte Carlo learning proof of concept:
    - [Implementation](https://github.com/da-roth/NeuronalNetworkTensorflowFramework/blob/main/src/Examples/MonteCarloLearning/Implementation_Multilevel_POC.ipynb)
    - [Colab Implementation](https://github.com/da-roth/NeuronalNetworkTensorflowFramework/blob/main/src/Examples_Colab/MonteCarloLearning/Implementation_Multilevel_POC.ipynb)


## Multilevel Monte Carlo Learning

This repository has been extended to include an implementation of the methods presented in the preprint ["Multilevel Monte Carlo Learning"](https://arxiv.org/abs/2102.08734). The article focuses on the approximation of expected values of functional quantities on the solution of a stochastic differential equation (SDE) using multilevel Monte Carlo learning techniques.

The main contributions and features of this extension are:

- Implementation of the multilevel approach for neural network training, as described in the "Multilevel Monte Carlo Learning" article
- Reduction of variance in the training process by shifting computational workload to training neural nets at coarse levels
- A complexity theorem demonstrating the reduction of computational complexity using the multilevel idea
- Example outputs and Colab executables, as used in the article, demonstrating the effectiveness of the methods and generating the article's results

The code for this extension can be found in the `multilevelmontecarlolearning` directory. Additionally, Colab executables and all output .csv used to calculate the results from the "Multilevel Monte Carlo Learning" article are available in the `multilevelmontecarlolearning/numericalresults` directory.

While, the fast-computable proof-of-concept [example](https://github.com/da-roth/NeuronalNetworkTensorflowFramework/blob/main/src/Examples/MonteCarloLearning/Implementation_Multilevel_POC.ipynb) was already mentioned above, the codes for the examples studied in the article can be found in the following files:

- Introductory example: single-level and multi-level
    - [Implementation](https://github.com/da-roth/NeuronalNetworkTensorflowFramework/blob/main/src/multilevelmontecarlolearning/IntroductoryExample/introductory_example.ipynb)
    - [Colab Implementation](https://github.com/da-roth/NeuronalNetworkTensorflowFramework/blob/main/src/multilevelmontecarlolearning/IntroductoryExample/introductory_example_Colab.ipynb)
    - [single-level results](https://github.com/da-roth/NeuronalNetworkTensorflowFramework/tree/main/src/multilevelmontecarlolearning/IntroductoryExample/single-introductory-outputs)
    - [multi-level results](https://github.com/da-roth/NeuronalNetworkTensorflowFramework/tree/main/src/multilevelmontecarlolearning/IntroductoryExample/multi-introductory-outputs)
  
- Multilevel example
    - [Implementation](https://github.com/da-roth/NeuronalNetworkTensorflowFramework/blob/main/src/multilevelmontecarlolearning/MultilevelExample/individual_steps_example.ipynb)
    - [Colab Implementation](https://github.com/da-roth/NeuronalNetworkTensorflowFramework/blob/main/src/multilevelmontecarlolearning/MultilevelExample/individual_steps_example_Colab.ipynb)
    - [results](https://github.com/da-roth/NeuronalNetworkTensorflowFramework/tree/main/src/multilevelmontecarlolearning/MultilevelExample/multilevel-example-outputs)


By exploring these examples, we hope to facilitate a deeper understanding of the techniques and results presented in the article, enabling you to apply the multilevel Monte Carlo learning approach to your own projects effectively.


## Contributing

We welcome contributions and suggestions to improve the framework. Please feel free to open an issue or submit a pull request with your proposed changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
