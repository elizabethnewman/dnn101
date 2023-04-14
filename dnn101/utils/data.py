import torch
import math


class DNN101Data:
    def __init__(self, *args, seed: int = 42, **kwargs):
        self.seed = seed

    def generate_data(self, n_samples=100):
        raise NotImplementedError

    def split_data(self, x, y, n_train=0.9, n_val=0.1):
        if not isinstance(n_train, int):
            n_train = math.floor(n_train * x.shape[0])

        if not isinstance(n_val, int):
            # default: 10% of training
            n_val = math.floor(n_val * n_train)
            n_train = n_train - n_val

        torch.manual_seed(self.seed)
        idx = torch.randperm(x.shape[0])  # shuffle data (not that important)

        x_train, y_train = x[idx[:n_train]], y[idx[:n_train]]
        x_val, y_val = x[idx[n_train:n_train + n_val]], y[idx[n_train:n_train + n_val]]
        x_test, y_test = x[idx[n_train + n_val:]], y[idx[n_train + n_val:]]

        return (x_train, y_train), (x_val, y_val), (x_test, y_test)

    def plot_data(self, *args):
        raise NotImplementedError

    def plot_prediction(self, net, *args):
        raise NotImplementedError
