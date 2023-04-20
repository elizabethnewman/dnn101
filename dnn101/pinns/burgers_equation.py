
import torch
from dnn101.utils import DNN101Data
from typing import Callable
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from hessQuik.utils import peaks
import sklearn.datasets as datasets
from copy import deepcopy
from dnn101.utils import DNN101Data
from dnn101.pinns import PDEDomain, PDEDomainBox, DNN101DataPINNHeatEquation1D, HeatEquation1DPINN



class BurgersEquation1DPINN(HeatEquation1DPINN):

    def __init__(self, net):
        super(BurgersEquation1DPINN, self).__init__(net)
        self.c = torch.nn.Parameter(torch.tensor(1))

    def forward(self, xt_int, f, xt_init, g, xt_bound, b):
        # evaluate initial condition
        n_init = xt_init.shape[0]
        pred_init = self.net(xt_init)
        loss_init = (1 / n_init) * torch.norm(pred_init - g) ** 2

        # evaluate boundary condition
        n_b = xt_bound.shape[0]
        pred_bound = self.net(xt_bound)
        loss_bound = (0.5 / n_b) * (torch.norm(pred_bound - b) ** 2)

        # evaluate physics and ensure we can differentiate
        n_int = xt_int.shape[0]
        x = xt_int[:, 0:1]
        t = xt_int[:, 1:2]
        x.requires_grad = True
        t.requires_grad = True
        pred_f = self.pde(x, t)
        loss_f = (0.5 / n_int) * torch.norm(pred_f - f) ** 2

        return loss_init, loss_bound, loss_f

    def pde(self, x, t):

        u = self.net(torch.cat((x, t), dim=1))

        u_t = torch.autograd.grad(u, t,
                                  grad_outputs=torch.ones_like(u),
                                  retain_graph=True,
                                  create_graph=True)[0]

        u_x = torch.autograd.grad(u, x,
                                  grad_outputs=torch.ones_like(u),
                                  retain_graph=True,
                                  create_graph=True)[0]

        u_xx = torch.autograd.grad(u_x, x,
                                   grad_outputs=torch.ones_like(u),
                                   retain_graph=True,
                                   create_graph=True)[0]

        return u_t + u * u_x - self.c * u_xx


class DNN101DataPINNBurgersEquation1D(DNN101DataPINNHeatEquation1D):
    def __init__(self, f: Callable, g: Callable, g_init: Callable, domain: PDEDomain = PDEDomainBox((-1, 1, 0, 10)), u_true: Callable = None):
        super(DNN101DataPINNBurgersEquation1D, self).__init__(f, g, g_init, domain=domain, u_true=u_true)


def pde_libraryBurgersEquation1D(fctn_num=0):

    if fctn_num == 0:
        p = torch.pi
        c = 0.1 / math.pi
        u_true = lambda xt: torch.exp(-xt[:, 1]) * torch.sin(p * xt[:, 0])
        pde = {'eqn': 'du/dt + u * du/dx - c * d2u/dx2 = f; u_{bd} = g; u_{init} = g_init',
               'c': c,
               'u_true': u_true,
               'f': lambda xt: -u_true(xt) + p * torch.exp(-xt[:, 1]) * torch.cos(p * xt[:, 0]) * u_true(xt) + c * p ** 2 * u_true(xt),
               'g': lambda xt: u_true(xt),
               'g_init': lambda xt: torch.sin(torch.pi * xt[:, 0])}

    elif fctn_num == 1:
        c = 0.1 / math.pi
        pde = {'eqn': 'du/dt + u * du/dx - c * d2u/dx2 = f; u_{bd} = g; u_{init} = g_init',
               'c': c,
               'u_true': None,
               'f': lambda xt: 0 * xt[:, 0],
               'g': lambda xt: 0 * xt[:, 0],  # Dirichlet
               'g_init': lambda xt: torch.sin(torch.pi * xt[:, 0])}
    else:
        raise ValueError('fctn_num = 0 or 1 for Heat Equation1D')

    return pde


if __name__ == "__main__":
    import torch.nn as nn
    # from dnn101.pinns.pde_data import plot_data

    pde_setup = pde_libraryBurgersEquation1D(1)
    f_true = pde_setup['f']
    g_true = pde_setup['g']
    g_init = pde_setup['g_init']
    u_true = pde_setup['u_true']

    pde = DNN101DataPINNBurgersEquation1D(f_true, g_true, g_init, domain=PDEDomainBox((-1, 1, 0, 10)), u_true=u_true)
    data_train, data_val, data_test = pde.generate_data()
    pde.plot_data(data_train, data_val, data_test)
    plt.show()

    net2D = nn.Sequential(nn.Linear(2, 10),
                          nn.Tanh(),
                          nn.Linear(10, 1))

    # plt.subplot(1, 3, 1)
    # pde.plot_prediction(pde.u_true)
    # plt.title('true')

    plt.subplot(1, 3, 2)
    pde.plot_prediction(net2D)
    plt.title('approx')

    # plt.subplot(1, 3, 3)
    # pde.plot_prediction(lambda x: torch.abs(net2D(x).view(-1) - pde.u_true(x).view(-1)))
    # plt.title('abs. diff.')
    plt.show()


    for t in range(0, 10, 2):
        pde.plot_slice(net2D, t, label=str(t))

    plt.title('slices')
    plt.legend()
    plt.show()

    # data_train, data_val, data_test = pde.generate_data()
    # plot_data(data_train, data_val, data_test)
    # plt.show()
    #
    # net2D = nn.Sequential(nn.Linear(2, 10),
    #                       nn.Tanh(),
    #                       nn.Linear(10, 1))
    # pde.plot_prediction2(net2D)
    # plt.show()
    #
    # pde.plot_slice(0, net2D)
    # plt.show()
