# Neural Network Regression Framework with Monte Carlo Learning

This repository contains a flexible neural network framework for regression-based training tasks, focusing on the exploration and understanding of Monte Carlo learning techniques. The primary goal is to investigate the impact of using sampled or approximated results as labels for randomly selected inputs during training. By leveraging the Monte Carlo approach, the framework can generate an unlimited amount of data, allowing neural networks to be evaluated on a continuous range of inputs.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Overview

The project is divided into two main components:
1. Development of a flexible framework for regression-based neural network training tasks using two primary classes of data input/generation methods:
   - General file inputs for precomputed data usage
   - Individual functions for on-the-fly data generation during training
2. Investigation of neural networks' properties with respect to Monte Carlo learning techniques.

## Installation

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

## Usage

After installing the required dependencies, you can run the example files in the `src/Examples` directory. The main framework is located in the `montecarlolearning` directory.

To use the framework in your own projects, simply import the required modules and functions.

## Examples

Several examples have been provided in the `src/Examples` directory, demonstrating the framework's capabilities and usage. A Google Colab executable link with all examples is also available [here](https://github.com/da-roth/NeuronalNetworkTensorflowFramework/blob/main/src/Examples_Colab/CumulativeDensitiyFunction/ImplementationTogether.ipynb).

Refer to the [documentation](https://github.com/da-roth/NeuronalNetworkTensorflowFramework/blob/main/src/documentation.pdf) for detailed explanations of the examples and code.

## Contributing

We welcome contributions and suggestions to improve the framework. Please feel free to open an issue or submit a pull request with your proposed changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


## Extension: Multilevel Monte Carlo Learning

This repository has been extended to include an implementation of the methods presented in the preprint "Multilevel Monte Carlo Learning", which can be found [here](https://arxiv.org/abs/2102.08734). The article focuses on the approximation of expected values of functional quantities on the solution of a stochastic differential equation (SDE) using multilevel Monte Carlo learning techniques.

The main contributions and features of this extension are:

- Implementation of the multilevel approach for neural network training, as described in the "Multilevel Monte Carlo Learning" article
- Reduction of variance in the training process by shifting computational workload to training neural nets at coarse levels
- A complexity theorem demonstrating the reduction of computational complexity using the multilevel idea
- Example outputs and Colab executables, as used in the article, demonstrating the effectiveness of the methods and generating the article's results

The code for this extension can be found in the `multilevelmontecarlolearning` directory. Additionally, Colab executables and all output .csv used to calculate the results from the "Multilevel Monte Carlo Learning" article are available in the `multilevelmontecarlolearning/numericalresults` directory.

Furthermore, we have included a fast-computable proof-of-concept [example](https://github.com/da-roth/NeuronalNetworkTensorflowFramework/blob/main/multilevelmontecarlolearning/proof-of-concept.ipynb) alongside the [examples](https://github.com/da-roth/NeuronalNetworkTensorflowFramework/blob/main/multilevelmontecarlolearning/examples-article.ipynb) from the article. By exploring these examples, we hope to facilitate a deeper understanding of the techniques and results presented in the article, enabling you to apply the multilevel Monte Carlo learning approach to your own projects effectively.
