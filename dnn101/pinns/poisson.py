import torch
from dnn101.utils import DNN101Data
from typing import Callable
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from hessQuik.utils import peaks
import sklearn.datasets as datasets
from copy import deepcopy
from dnn101.pinns import DNN101DataPINN
from dnn101.pinns import PDEDomain, PDEDomainBox


class PoissonEquation2DPINN(torch.nn.Module):

    def __init__(self, net):
        super(PoissonEquation2DPINN, self).__init__()
        self.net = net

    def forward(self, data):
        (xy_int, f), (xy_bd, g) = data

        # number of interior and boundary points
        n_int = xy_int.shape[0]
        n_bd = xy_bd.shape[0]

        # evaluate boundary condition
        pred_bound = self.net(xy_bd)
        loss_bound = (0.5 / n_bd) * torch.norm(pred_bound.reshape(-1) - g.reshape(-1)) ** 2

        # evaluate physics and ensure we can differentiate
        x = xy_int[:, 0:1]
        y = xy_int[:, 1:2]
        x.requires_grad = True
        y.requires_grad = True
        pred_f = self.pde(x, y)
        loss_f = (0.5 / n_int) * torch.norm(pred_f.reshape(-1) - f.reshape(-1)) ** 2

        return loss_bound, loss_f

    def pde(self, x, y):

        u = self.net(torch.cat((x, y), dim=1))

        u_x = torch.autograd.grad(u, x,
                                  grad_outputs=torch.ones_like(u),
                                  retain_graph=True,
                                  create_graph=True)[0]

        u_y = torch.autograd.grad(u, y,
                                  grad_outputs=torch.ones_like(u),
                                  retain_graph=True,
                                  create_graph=True)[0]

        u_xx = torch.autograd.grad(u_x, x,
                                   grad_outputs=torch.ones_like(u),
                                   retain_graph=True,
                                   create_graph=True)[0]

        u_yy = torch.autograd.grad(u_y, y,
                                   grad_outputs=torch.ones_like(u),
                                   retain_graph=True,
                                   create_graph=True)[0]

        return u_xx + u_yy


class DNN101DataPINNPoisson2D(DNN101DataPINN):
    def __init__(self, f: Callable, g: Callable, domain: PDEDomain = PDEDomainBox((-1, 1, -1, 1)), u_true: Callable = None):
        super(DNN101DataPINNPoisson2D, self).__init__()
        self.f = f
        self.g = g
        self.domain = domain
        self.domain_labels = ('x1', 'x2')
        self.u_true = u_true

    def generate_data(self, n_train=80, n_val=10, n_test=10, p_bd=0.2):
        # p_bd is fraction of (n_train + n_val + n_test), the ratio of boundary to interior training points

        # points on the interior
        n_int = n_train + n_val + n_test
        xy_int = self.domain.generate_interior_points(n_int)

        # true right-hand side (source term)
        f = self.f(xy_int).view(-1, 1)

        # split into training and testing
        (x_int_train, f_train), (x_int_val, f_val), (x_int_test, f_test) = self.split_data(xy_int, f, n_train, n_val)

        # points on the boundary
        n_bd = math.floor(p_bd * n_int)
        xy_bd = self.domain.generate_boundary_points(n_bd)
        g = self.g(xy_bd).view(-1, 1)

        n_tot = xy_bd.shape[0]
        p_train = n_train / n_int
        p_val = n_val / n_int

        (x_bd_train, g_train), (x_bd_val, g_val), (x_bd_test, g_test) = self.split_data(xy_bd, g, max(10, math.floor(p_train * n_tot)), max(10, math.floor(p_val * n_val)))

        return ((x_int_train, f_train), (x_bd_train, g_train)), ((x_int_val, f_val), (x_bd_val, g_val)), ((x_int_test, f_test), (x_bd_test, g_test))


def pde_library_Poisson2D(fctn_num=0):

    if fctn_num == 0:
        a, b = 2.0, 4.0
        u_true = lambda x: a * x[:, 0] ** 2 + b * x[:, 1] ** 2
        pde = {'eqn': 'd^2u/dx^2 + d^2u/dy^2 = f; u_{bd} = g',
               'a': a,
               'b': b,
               'u_true': u_true,
               'f': lambda x: 2 * a + 2 * b + 0 * x[:, 0],
               'g': u_true}
    elif fctn_num == 1:
        a, b = 4.0, 2.0
        u_true = lambda x: torch.sin(a * torch.pi * x[:, 0]) * torch.sin(b * torch.pi * x[:, 1])
        pde = {'eqn': 'd^2u/dx^2 + d^2u/dy^2 = f; u_{bd} = g',
               'a': a,
               'b': b,
               'u_true': u_true,
               'f': lambda x: -(a ** 2 + b ** 2) * (torch.pi ** 2) * u_true(x),
               'g': u_true}
    else:
        raise ValueError('fctn_num = 0, 1, 2 for Poisson')

    return pde


if __name__ == "__main__":
    import torch.nn as nn
    # from dnn101.pinns.pde_data import plot_data

    pde_setup = pde_libraryPoisson2D(1)
    f_true = pde_setup['f']
    g_true = pde_setup['g']
    u_true = pde_setup['u_true']

    pde = DNN101DataPINNPoisson2D(f_true, g_true, u_true=u_true)

    data_train, data_val, data_test = pde.generate_data()
    pde.plot_data(data_train, data_val, data_test)
    plt.show()

    net2D = nn.Sequential(nn.Linear(2, 10),
                          nn.Tanh(),
                          nn.Linear(10, 1))

    plt.subplot(1, 3, 1)
    pde.plot_prediction(pde.u_true)
    plt.title('true')

    plt.subplot(1, 3, 2)
    pde.plot_prediction(net2D)
    plt.title('approx')

    plt.subplot(1, 3, 3)
    pde.plot_prediction(lambda x: torch.abs(net2D(x).view(-1) - pde.u_true(x).view(-1)))
    plt.title('abs. diff.')
    plt.show()
