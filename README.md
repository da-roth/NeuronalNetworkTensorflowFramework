# Neural Network Regression Framework with Monte Carlo Learning

This framework is designed for regression-based neural network training tasks, with a focus on Monte Carlo learning. The aim is to provide a flexible and customizable environment for experimenting with neural networks and their application to various types of data input/generation methods. The repository contains a series of examples that demonstrate the core concepts, with increasing levels of complexity.

In addition to the provided examples, there is an [introduction PDF](https://github.com/da-roth/NeuronalNetworkTensorflowFramework/blob/main/documentation.pdf) that delves into the mathematical background of Monte Carlo learning.

## Table of Contents

- [Overview](#overview)
- [Examples](#examples)
- [Installation](#installation)
- [Usage](#usage)
- [Multilevel Monte Carlo Learning](#multilevel-monte-carlo-learning)
- [Related Works](#related-works)
- [Future Work](#future-work)
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

The supported functions, specifications and further details are provided in the [documentation](https://github.com/da-roth/NeuronalNetworkTensorflowFramework/blob/main/documentation.pdf).

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


By exploring these examples, the goal is to facilitate a deeper understanding of the techniques and results presented in the article, enabling you to apply the multilevel Monte Carlo learning approach to your own projects effectively.

## Related Works

This repository is inspired by and builds upon various works from the literature:

- **Multilevel Monte Carlo Learning**: Gerstner, T., Harrach, B., Roth, D., & Simon, M. (2021). [Multilevel Monte Carlo Learning](https://arxiv.org/abs/2102.08734). *arXiv preprint arXiv:2102.08734*. This article introduces a novel approach to solving partial/stochastic differential equations (PDEs/SDEs) using a combination of deep learning and multilevel Monte Carlo methods. The authors use multiple neural networks to learn the solution of the SDEs level estimators.

- **Solving Stochastic Differential Equations and Kolmogorov Equations by Means of Deep Learning**: Beck, C., Becker, S., Grohs, P., Jaafari, N., & Jentzen, A. (2018). [Solving Stochastic Differential Equations and Kolmogorov Equations by Means of Deep Learning](https://arxiv.org/abs/1806.00421). *arXiv preprint arXiv:1806.00421*. The authors propose a method based on deep neural networks to solve both stochastic differential equations. This work introduced the underlying idea of using deep learning to solve PDEs.

- **Differential Machine Learning**: The code structure of classes used in this framework was adapted from the [GitHub repository](https://github.com/differential-machine-learning/notebooks) by Huge and Savine, which provides a collection of Jupyter notebooks illustrating various aspects of differential machine learning. The code was modified and extended to suit the specific needs of this project.

## Future Work

This section outlines the ideas and plans for future development of the project. Contributions and suggestions from the community are highly encouraged. To contribute, please open an issue or submit a pull request.

### Training Settings and Neural Network Architecture Settings

The framework can be enhanced by incorporating a greater variety of training and neural network architecture settings:

1. **More training settings**: Exploring different training settings, such as varying learning rate schedules, optimization algorithms, and loss functions, can lead to improved model performance.
2. **Neural network architecture settings**: Supporting a wider range of neural network architectures allows the framework to be more adaptable to different use cases and preferences. This includes the option to use custom or pretrained models, as well as varying layer configurations.

### Differential Machine Learning

Integrating differential machine learning into the framework by:

1. **TrainingDataGenerator**: Modifying the TrainingDataGenerator to accept pathwise sensitivity outputs, which would enable differential information to be easily incorporated into the training data.
2. **Loss functions and evaluation metrics**: Implementing differential-aware loss functions and evaluation metrics could improve the training process and performance measurement for models working with differential data.
3. **Exploring TensorFlow's automatic differentiation**

Any feedback, suggestions, or contributions to help make this project even better are greatly appreciated. Thank you for your support!

## Contributing

I welcome contributions and suggestions to improve the framework. Please feel free to open an issue or submit a pull request with your proposed changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
