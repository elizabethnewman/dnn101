## Organization

Each directory has focuses on a different task for which deep learning can be used (and used well!).  The goal is to learn something new with each task.  Each directory contains at least one corresponding Google Colab notebook and methods to generate and visualize synthetic data.

### Notebooks for Newcomers
* [regression](https://github.com/elizabethnewman/dnn101/tree/main/dnn101/regression): one- and two-dimensional function approximation tasks
    * Goals: construct a neural network from data to done for function approximation (a good place to start for beginners)
* [classification](https://github.com/elizabethnewman/dnn101/tree/main/dnn101/classification): classifying two-dimensional data
* [cnn](https://github.com/elizabethnewman/dnn101/tree/main/dnn101/cnn): convolutional neural networks for image classification

### Notebooks with More Sophisticated Implementations
* [pinns](https://github.com/elizabethnewman/dnn101/tree/main/dnn101/pinns): physics-informed neural networks with toy problems
    *  Goals: learn how to code PINNs using automatic differentation and how to use the L-BFGS optimizers in PyTorch (a good place to start if you have some knowledge of PyTorch and DNNs already)
* [dynamics](https://github.com/elizabethnewman/dnn101/tree/main/dnn101/dynamics): residual neural networks
    * Goals: learn how to implement new PyTorch layers and explore DNN stability (this is a more exploratory task)
    
### Utilities and Testing
* [utils](https://github.com/elizabethnewman/dnn101/tree/main/dnn101/utils): helpful functionality for all directories
