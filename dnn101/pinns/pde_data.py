import matplotlib.pyplot as plt
import torch
from typing import Callable
from dnn101.utils import DNN101Data
from dnn101.pinns import PDEDomain, PDEDomainBox


class DNN101DataPINN(DNN101Data):
    def __init__(self, *args, domain: PDEDomain = PDEDomainBox((-1, 1, -1, 1)), u_true: Callable = None):
        super(DNN101DataPINN, self).__init__()
        self.domain = domain
        self.domain_labels = ('x1', 'x2')
        self.u_true = u_true

    def plot_data(self, *args, labels=('train', 'val', 'test'),
                  regions=(': int', ': bd', ': init', ': init_deriv'),
                  markers=('o', 's', '^')):
        # args:  ((x_set1, y_set1), (x_bd_set1, y_bd_set1), ...), ((x_set2, y_set2), (x_bd_set2, y_bd_set2, ...)),...
        color_order = plt.rcParams['axes.prop_cycle'].by_key()['color']

        # train, val, test loop
        for count, a in enumerate(args):

            # int, bd, ... loop
            for i, b in enumerate(a):
                x = b[0]
                y = b[1]
                if x is not None and y is not None:
                    plt.scatter(x[:, 0], x[:, 1], color=color_order[i], marker=markers[count], label=labels[count] + regions[i])

        plt.xlabel(self.domain_labels[0])
        plt.ylabel(self.domain_labels[1])
        plt.legend()

    def plot_prediction(self, *args):
        f = args[0]  # a function or network
        with torch.no_grad():
            x1_grid, x2_grid = self.domain.generate_grid2D()
            x_grid = torch.cat((x1_grid.reshape(-1, 1), x2_grid.reshape(-1, 1)), dim=1)
            plt.contourf(x1_grid, x2_grid, f(x_grid).reshape(x1_grid.shape).detach())
            plt.xlabel(self.domain_labels[0])
            plt.ylabel(self.domain_labels[1])

    def plot_slice(self, f, t, label='true'):
        x1_grid = torch.linspace(self.domain.domain[0], self.domain.domain[1], 50)
        x_grid = torch.cat((x1_grid.reshape(-1, 1), t * torch.ones_like(x1_grid).reshape(-1, 1)), dim=1)

        u = f(x_grid)
        plt.plot(x1_grid.view(-1), u.view(-1).detach(), label=label)
        plt.xlim(self.domain.domain[0], self.domain.domain[1])
