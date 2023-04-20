import torch
from dnn101.utils.data import DNN101Data
from typing import Callable
import matplotlib as mpl
import matplotlib.pyplot as plt
from hessQuik.utils import peaks
import sklearn.datasets as datasets
from copy import deepcopy


class DNN101DataClassification2D(DNN101Data):
    def __init__(self, f: Callable, n_classes: int = 2, domain: tuple = (-3, 3, -3, 3), noise_level: float = 1e-2):
        super(DNN101DataClassification2D, self).__init__()
        self.f = f
        self.n_classes = n_classes
        self.domain = domain
        self.noise_level = noise_level  # TODO: use this input

        # choose class levels
        x1_grid, x2_grid = torch.meshgrid(torch.linspace(self.domain[0], self.domain[1], 100),
                                          torch.linspace(self.domain[2], self.domain[3], 100), indexing='ij')
        f_grid = self.f(torch.cat((x1_grid.reshape(-1, 1), x2_grid.reshape(-1, 1)), dim=1))
        min_F, max_F = f_grid.min(), f_grid.max()
        h = (max_F - min_F) / n_classes
        self.cutoffs = min_F + h * torch.arange(1, n_classes)

    def _get_labels(self, f_pts):
        if f_pts.ndim < 2 or f_pts.shape[1] == 1:
            y = torch.zeros(f_pts.numel(), dtype=torch.int64)
            for i in range(1, self.n_classes - 1):
                y[(f_pts.view(-1) > self.cutoffs[i - 1]) * (f_pts.view(-1) <= self.cutoffs[i])] = i
            y[(f_pts.view(-1) > self.cutoffs[-1])] = self.n_classes - 1
        else:
            y = f_pts.argmax(dim=1).view(-1)

        return y.view(-1)

    def generate_data(self, n_samples=100):
        x1 = self.domain[0] + (self.domain[1] - self.domain[0]) * torch.rand(n_samples)
        x2 = self.domain[2] + (self.domain[3] - self.domain[2]) * torch.rand(n_samples)
        x = torch.cat((x1.reshape(-1, 1), x2.reshape(-1, 1)), dim=1)

        f_true = self.f(x)
        f_true += self.noise_level * torch.randn_like(f_true)
        y = self._get_labels(f_true)

        return x, y

    def plot_data(self, *args, labels=('train', 'val', 'test'), markers=('o', 's', '^')):
        # args:  ((x_set1, y_set1), (x_bd_set1, y_bd_set1), ...), ((x_set2, y_set2), (x_bd_set2, y_bd_set2, ...)),...
        tmp = plt.rcParams['axes.prop_cycle'].by_key()['color']
        cmap = mpl.colors.ListedColormap(tmp[:self.n_classes])
        # train, val, test loop
        for count, a in enumerate(args):
                x = a[0]
                y = a[1]
                if x is not None and y is not None:
                    plt.scatter(x[:, 0], x[:, 1], None, y, cmap=cmap, marker=markers[count], label=labels[count])

        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend()

    def plot_prediction(self, f, x_pts=None, y_pts=None):
        with torch.no_grad():
            x1_grid, x2_grid = torch.meshgrid(torch.linspace(self.domain[0], self.domain[1], 200),
                                              torch.linspace(self.domain[2], self.domain[3], 200), indexing='ij')
            x_grid = torch.cat((x1_grid.reshape(-1, 1), x2_grid.reshape(-1, 1)), dim=1)

            f_grid = f(x_grid)
            if f_grid.ndim <= 1 or f_grid.shape[1] == 1:
                y_grid = self._get_labels(f_grid)
            else:
                y_grid = f_grid.argmax(dim=1)
            tmp = plt.rcParams['axes.prop_cycle'].by_key()['color']
            cmap = mpl.colors.ListedColormap(tmp[:self.n_classes])
            plt.contourf(x1_grid, x2_grid, y_grid.reshape(x1_grid.shape), cmap=cmap, alpha=0.5)

            if x_pts is not None:
                plt.scatter(x_pts[:, 0], x_pts[:, 1], None, y_pts, cmap=cmap, label='points')

            plt.xlabel('x1')
            plt.ylabel('x2')

    def plot_propagated_features(self, net, z, y):
        z1_grid, z2_grid = torch.meshgrid(torch.linspace(self.domain[0], self.domain[1], 200),
                                          torch.linspace(self.domain[2], self.domain[3], 200), indexing='ij')
        z_grid = torch.cat((z1_grid.reshape(-1, 1), z2_grid.reshape(-1, 1)), dim=1)

        k = net[-1].weight.shape[1] - z_grid.shape[1]
        tmp = torch.cat((z_grid, torch.zeros(z_grid.shape[0], k)), dim=1)
        z_labels = self._get_labels(net[-1](tmp))

        tmp = plt.rcParams['axes.prop_cycle'].by_key()['color']
        cmap = mpl.colors.ListedColormap(tmp[:self.n_classes])
        plt.contourf(z1_grid, z2_grid, z_labels.reshape(z1_grid.shape), cmap=cmap, alpha=0.5)

        # propagate through all but final layer (output features)
        for i in range(len(net) - 1):
            z = net[i](z)

        # plot first two columns
        for i in range(self.n_classes):
            plt.scatter(z[y == i, 0].detach(), z[y == i, 1].detach())


class DNN101DataClassificationSKLearn(DNN101Data):
    def __init__(self, name: str = 'blobs', **kwargs):
        super(DNN101DataClassificationSKLearn, self).__init__()
        self.name = name
        self.domain = None
        self.n_classes = None
        self.kwargs = kwargs

    def generate_data(self, n_samples=100):
        if self.name == 'circles':
            (x, y) = datasets.make_circles(n_samples=n_samples, **self.kwargs)
        elif self.name == 'moons':
            (x, y) = datasets.make_moons(n_samples=n_samples, **self.kwargs)
        else:
            (x, y) = datasets.make_blobs(n_samples=n_samples, **self.kwargs)

        a = torch.from_numpy(x.min(axis=0)) - 0.5
        b = torch.from_numpy(x.max(axis=0)) + 0.5
        c = torch.zeros(a.numel() + b.numel())
        c[0::2] = a
        c[1::2] = b
        self.domain = list(c)

        # convert to tensor
        x = torch.from_numpy(x).to(torch.float32)
        y = torch.from_numpy(y).to(torch.int64)
        self.n_classes = len(y.unique())

        return x, y

    def _get_labels(self, f_pts):
        return f_pts.argmax(dim=1).view(-1)

    def plot_data(self, *args, labels=('train', 'val', 'test'), markers=('o', 's', '^')):
        # args:  ((x_set1, y_set1), (x_bd_set1, y_bd_set1), ...), ((x_set2, y_set2), (x_bd_set2, y_bd_set2, ...)),...
        tmp = plt.rcParams['axes.prop_cycle'].by_key()['color']
        cmap = mpl.colors.ListedColormap(tmp[:self.n_classes])
        # train, val, test loop
        for count, a in enumerate(args):
                x = a[0]
                y = a[1]
                if x is not None and y is not None:
                    plt.scatter(x[:, 0], x[:, 1], None, y, cmap=cmap, marker=markers[count], label=labels[count])

        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend()

    def plot_prediction(self, f, x_pts=None, y_pts=None):
        with torch.no_grad():
            x1_grid, x2_grid = torch.meshgrid(torch.linspace(self.domain[0], self.domain[1], 200),
                                              torch.linspace(self.domain[2], self.domain[3], 200), indexing='ij')
            x_grid = torch.cat((x1_grid.reshape(-1, 1), x2_grid.reshape(-1, 1)), dim=1)

            f_grid = f(x_grid)
            if f_grid.ndim <= 1 or f_grid.shape[1] == 1:
                y_grid = self._get_labels(f_grid)
            else:
                y_grid = f_grid.argmax(dim=1)
            tmp = plt.rcParams['axes.prop_cycle'].by_key()['color']
            cmap = mpl.colors.ListedColormap(tmp[:self.n_classes])
            plt.contourf(x1_grid, x2_grid, y_grid.reshape(x1_grid.shape), cmap=cmap, alpha=0.5)

            if x_pts is not None:
                plt.scatter(x_pts[:, 0], x_pts[:, 1], None, y_pts, cmap=cmap, label='points')

            plt.xlabel('x1')
            plt.ylabel('x2')


if __name__ == "__main__":
    import torch.nn as nn
    import matplotlib as mpl

    mpl.rcParams['figure.figsize'] = (8, 6)
    mpl.rcParams['lines.linewidth'] = 8
    mpl.rcParams['lines.markersize'] = 10
    mpl.rcParams['font.size'] = 10

    net2D = nn.Sequential(nn.Linear(2, 10),
                          nn.Tanh(),
                          nn.Linear(10, 1))

    f = lambda x: torch.sin(x[:, 0]) + torch.cos(x[:, 1])

    data2D = DNN101DataClassification2D(f, 3)
    x, y = data2D.generate_data(n_samples=2000)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = data2D.split_data(x, y)

    data2D.plot_data((x_train, y_train), (x_val, y_val), (x_test, y_test))
    plt.show()

    plt.subplot(1, 2, 1)
    data2D.plot_prediction(data2D.f, x_test, y_test)
    plt.title('true')

    plt.subplot(1, 2, 2)
    data2D.plot_prediction(net2D, x_test, y_test)
    plt.title('prediction')
    plt.show()

    data2D.plot_propagated_features(net2D, x_test, y_test)
    plt.show()

    data2D = DNN101DataClassificationSKLearn('moons')
    x, y = data2D.generate_data(n_samples=2000)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = data2D.split_data(x, y, n_train=1000)

    data2D.plot_data((x_train, y_train), (x_val, y_val), (x_test, y_test))
    plt.show()

    data2D.plot_prediction(net2D, x_test, y_test)
    plt.show()
