# dnn101
A hands-on tutorial of deep neural networks for mathematicians

# Introduction

Deep learning has revolutionized data science and has given state-of-the-art results across applications that require high-dimensional function approximation and classification. This tutorial will help you understand the basics of deep neural networks using hands-on Google Colab notebooks.

# Installation

```console
python -m pip install git+https://github.com/elizabethnewman/dnn101.git
```

# Getting Started

Let's generate a regression example!  First, we load the necessary packages and generate the data.

```python
from dnn101.regression import DNN101DataRegression1D
import torch
import matplotlib.pyplot as plt

# data parameters
n_train = 1000                      # number of training points
n_val   = 100                       # number of validation points
n_test  = 100                       # number of test points
sigma   = 0.2                       # noise level
f       = lambda x: torch.sin(x)    # function to approximate
domain  = (-3, 3)                   # function domain

# create data set
dataset = DNN101DataRegression1D(f, domain, noise_level=sigma)

# generate data
x, y = dataset.generate_data(n_train + n_val + n_test)

# split into training, validation, and test sets
(x_train, y_train), (x_val, y_val), (x_test, y_test) = dataset.split_data(x, y, n_train=n_train, n_val=n_val)

# plot!
dataset.plot_data(x_train, y_train, x_val, y_val, x_test, y_test)
plt.show()
```
![Regression Data](docs/figs/getting_started_regression_data.png)

# Organization

Each directory has focuses on a different task for which deep learning can be used (and used well!).  The goal is to learn something new with each task.  Each directory contains at least one corresponding Google Colab notebook and methods to generate and visualize synthetic data.

* ```regression```: one- and two-dimensional function approximation tasks
    * Goals: construct a neural network from data to done for function approximation (a good place to start for beginners)
* ```classification```: classifying two-dimensional data
    * Goals: construct a neural network from data to done for classification (a good place to start for beginners)
* ```cnn```: convolutional neural networks for image classification
    * Goals: learn how to code a CNN and how to use GPUs on Google Colab (some details hidden)
* ```pinns```: physics-informed neural networks with toy problems
    *  Goals: learn how to code PINNs using automatic differentation and how to use the L-BFGS optimizers in PyTorch (a good place to start if you have some knowledge of PyTorch and DNNs already)
* ```dynamics```: residual neural networks
    * Goals: learn how to implement new PyTorch layers and explore DNN stability (this is a more exploratory task)
* ```utils```: helpful functionality for all directories


# Cite

TBD
