import torch
from typing import Callable
import matplotlib.pyplot as plt
from dnn101.utils import DNN101Data


class DNN101DataRegression(DNN101Data):
    def __init__(self, f: Callable, domain: tuple = (-3, 3), noise_level: float = 1e-2):
        super(DNN101DataRegression, self).__init__()
        self.f = f
        self.domain = domain
        self.d = len(domain) // 2
        self.noise_level = noise_level

    def generate_data(self, n_samples=100):
        x = torch.empty(0)
        for i in range(self.d):
            x_tmp = self.domain[0] + (self.domain[1] - self.domain[0]) * torch.rand(n_samples)
            x = torch.cat((x, x_tmp.view(-1, 1)), dim=1)

        y_true = self.f(x).view(-1, 1)  # no noise
        y = y_true + self.noise_level * torch.randn(y_true.shape)

        return x, y


class DNN101DataRegression1D(DNN101DataRegression):
    def __init__(self, f: Callable, domain: tuple = (-3, 3), noise_level: float = 1e-2):
        super(DNN101DataRegression1D, self).__init__(f=f, domain=domain, noise_level=noise_level)

    def plot_data(self, *args, labels=('train', 'val', 'test'), markers=('o', 's', '^')):
        for count, i in enumerate(range(0, len(args), 2)):
            x = args[i]
            y = args[i + 1]
            if x is not None and y is not None:
                plt.plot(x, y, markers[count], label=labels[count])

        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()

    def plot_prediction(self, net, *args):
        with torch.no_grad():
            x_grid = torch.linspace(self.domain[0], self.domain[1], 100).view(-1, 1)
            plt.plot(x_grid, net(x_grid), '--', label='pred', color='g')
            plt.plot(x_grid, self.f(x_grid), label='true', color='b')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend()


class DNN101DataRegression2D(DNN101DataRegression):
    def __init__(self, f: Callable, domain: tuple = (-3, 3, -3, 3), noise_level: float = 1e-2):
        super(DNN101DataRegression2D, self).__init__(f=f, domain=domain, noise_level=noise_level)

    def plot_data(self, *args, labels=('train', 'val', 'test'), markers=('o', 's', '^')):
        for count, i in enumerate(range(0, len(args), 2)):
            x = args[i]
            y = args[i + 1]
            if x is not None and y is not None:
                plt.scatter(x[:, 0], x[:, 1], None, y, marker=markers[count], label=labels[count])

        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar()
        plt.legend()

    def plot_prediction(self, net, *args):
        with torch.no_grad():
            x1_grid, x2_grid = torch.meshgrid(torch.linspace(self.domain[0], self.domain[1], 50),
                                              torch.linspace(self.domain[2], self.domain[3], 50), indexing='ij')
            x_grid = torch.cat((x1_grid.reshape(-1, 1), x2_grid.reshape(-1, 1)), dim=1)

            plt.subplot(1, 3, 1)
            plt.contourf(x1_grid, x2_grid, self.f(x_grid).reshape(x1_grid.shape))
            plt.xlabel('x1')
            plt.ylabel('x2')
            plt.colorbar()
            plt.title('true')

            plt.subplot(1, 3, 2)
            plt.contourf(x1_grid, x2_grid, net(x_grid).reshape(x1_grid.shape))
            plt.xlabel('x1')
            plt.ylabel('x2')
            plt.colorbar()
            plt.title('approx')

            plt.subplot(1, 3, 3)
            plt.contourf(x1_grid, x2_grid,
                         torch.abs(net(x_grid).reshape(-1) - self.f(x_grid).reshape(-1)).reshape(x1_grid.shape))
            plt.xlabel('x1')
            plt.ylabel('x2')
            plt.colorbar()
            plt.title('abs. diff.')

    def plot_prediction3D(self, net):

        with torch.no_grad():
            x1_grid, x2_grid = torch.meshgrid(torch.linspace(self.domain[0], self.domain[1], 50),
                                              torch.linspace(self.domain[2], self.domain[3], 50), indexing='ij')
            x_grid = torch.cat((x1_grid.reshape(-1, 1), x2_grid.reshape(-1, 1)), dim=1)

            _, (ax1, ax2, ax3) = plt.subplots(1, 3, subplot_kw=dict(projection='3d'))

            ax1.plot_surface(x1_grid, x2_grid, self.f(x_grid).reshape(x1_grid.shape), cmap=mpl.cm.viridis)
            ax1.set_xlabel('x1')
            ax1.set_ylabel('x2')
            ax1.set_zlabel('f(x1, x2)')
            ax1.set_title('true')

            ax2.plot_surface(x1_grid, x2_grid, net(x_grid).reshape(x1_grid.shape), cmap=mpl.cm.viridis)
            ax2.set_xlabel('x1')
            ax2.set_ylabel('x2')
            ax2.set_zlabel('f(x1, x2)')
            ax2.set_title('approx')

            ax3.plot_surface(x1_grid, x2_grid,
                             torch.abs(net(x_grid).reshape(-1) - self.f(x_grid).reshape(-1)).reshape(x1_grid.shape),
                             cmap=mpl.cm.viridis)
            ax3.set_xlabel('x1')
            ax3.set_ylabel('x2')
            ax3.set_zlabel('f(x1, x2)')
            ax3.set_title('abs. diff.')


if __name__ == "__main__":
    import torch.nn as nn
    import matplotlib as mpl

    mpl.rcParams['figure.figsize'] = (8, 6)
    mpl.rcParams['lines.linewidth'] = 8
    mpl.rcParams['lines.markersize'] = 15
    mpl.rcParams['font.size'] = 10

    net1D = nn.Sequential(nn.Linear(1, 10),
                          nn.Tanh(),
                          nn.Linear(10, 1))

    f = lambda x: torch.sin(x)
    data1D = DNN101DataRegression1D(f, noise_level=1e-1)

    x, y = data1D.generate_data()
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = data1D.split_data(x, y)

    data1D.plot_data(x_train, y_train, x_val, y_val, x_test, y_test)
    plt.show()

    data1D.plot_prediction(net1D)
    plt.show()

    net2D = nn.Sequential(nn.Linear(2, 10),
                          nn.Tanh(),
                          nn.Linear(10, 1))

    f = lambda x: torch.sin(x[:, 0]) * (x[:, 1] + 1) + torch.cos(x[:, 1]) * x[:, 0]
    # f = function_library2D(1)

    data2D = DNN101DataRegression2D(f, noise_level=1e-1)

    n_train = 100
    n_val = 20
    n_test = 11
    x, y = data2D.generate_data(n_train + n_val + n_test)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = data2D.split_data(x, y, n_train=n_train, n_val=n_val)

    data2D.plot_data(x_train, y_train, x_val, y_val, x_test, y_test)
    plt.show()

    data2D.plot_prediction(net2D)
    plt.show()

    data2D.plot_prediction3D(net2D)
    plt.show()

