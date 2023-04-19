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
from dnn101.pinns import PDEDomain, PDEDomainBox, DNN101DataPINNHeatEquation1D


class WaveEquationPINN(torch.nn.Module):

    def __init__(self, net):
        super(WaveEquationPINN, self).__init__()

        self.net = net
        self.c = torch.nn.Parameter(torch.tensor(1))  # learnable parameter

    def forward(self, xt_int, f, xt_init, g, g_deriv):
        # evaluate initial condition
        n_init = xt_init.shape[0]
        pred_init = self.net(xt_init)
        loss_init = (0.5 / n_init) * torch.norm(pred_init - g) ** 2

        # evaluate initial conditions for derivative
        x = xt_init[:, 0:1]
        t = xt_init[:, 1:2]
        x.requires_grad = True
        t.requires_grad = True

        pred_init_deriv = self.Neumann_initial(x, t)
        loss_init_deriv = (0.5 / n_init) * torch.norm(pred_init_deriv - g_deriv) ** 2

        # evaluate physics and ensure we can differentiate
        n_int = xt_int.shape[0]
        x = xt_int[:, 0:1]
        t = xt_int[:, 1:2]
        x.requires_grad = True
        t.requires_grad = True
        pred_f = self.pde(x, t)
        loss_f = (0.5 / n_int) * torch.norm(pred_f - f) ** 2

        return loss_init, loss_init_deriv, loss_f

    def Neumann_initial(self, x, t):
        u = self.net(torch.cat((x, t), dim=1))

        u_t = torch.autograd.grad(u, t,
                                  grad_outputs=torch.ones_like(u),
                                  retain_graph=True,
                                  create_graph=True)[0]
        return u_t

    def pde(self, x, t):

        u = self.net(torch.cat((x, t), dim=1))

        u_t = torch.autograd.grad(u, t,
                                  grad_outputs=torch.ones_like(u),
                                  retain_graph=True,
                                  create_graph=True)[0]

        u_tt = torch.autograd.grad(u_t, t,
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

        return u_tt - (self.c ** 2) * u_xx


class DNN101DataPINNWaveEquation1D(DNN101DataPINN):
    def __init__(self, f: Callable, g: Callable, g_init: Callable, g_init_deriv: Callable, domain: PDEDomain = PDEDomainBox((-1, 1, -1, 1)), u_true: Callable = None):
        super(DNN101DataPINNWaveEquation1D, self).__init__(domain=domain, u_true=u_true)
        self.f = f
        self.g = g
        self.g_init = g_init
        self.g_init_deriv = g_init_deriv
        self.domain_labels = ('x', 't')

    def generate_data(self, n_train=80, n_val=10, n_test=10, p_bd=0.2, p_init=0.2):
        # p_bd is fraction of (n_train + n_val + n_test), the ratio of boundary to interior training points

        # points on the interior
        n_int = n_train + n_val + n_test
        xt_int = self.domain.generate_interior_points(n_int)

        # true right-hand side (source term)
        f = self.f(xt_int).view(-1, 1)

        # split into training and testing
        (x_int_train, f_train), (x_int_val, f_val), (x_int_test, f_test) = self.split_data(xt_int, f, n_train, n_val)

        # points on the boundary
        n_bd = math.floor(p_bd * n_int)
        xt_bd = self.domain.generate_boundary_points(n_bd)
        # xt_init = xt_bd[:xt_bd.shape[0] // 4]
        xt_bd = xt_bd[xt_bd.shape[0] // 2:]
        g = self.g(xt_bd).view(-1, 1)

        n_tot = xt_bd.shape[0]
        p_train = n_train / n_int
        p_val = n_val / n_int

        xt_bd = self.domain.generate_boundary_points(n_bd)
        g = self.g(xt_bd)
        xg_train, xg_val, xg_test = self.split_data(xt_bd, g, max(10, math.floor(p_train * n_tot)), max(10, math.floor(p_val * n_val)))

        # initial conditions
        xt_bd = self.domain.generate_boundary_points(n_bd)
        xt_init = xt_bd[:xt_bd.shape[0] // 4]
        g_init = self.g_init(xt_init).view(-1, 1)

        n_tot = xt_init.shape[0]
        p_train = n_train / n_int
        p_val = n_val / n_int
        xg_init_train, xg_init_val, xg_init_test = self.split_data(xt_init, g_init, max(10, math.floor(p_train * n_tot)), max(10, math.floor(p_val * n_val)))

        xt_bd = self.domain.generate_boundary_points(n_bd)
        xt_init_deriv = xt_bd[:xt_bd.shape[0] // 4]
        g_init_deriv = self.g_init_deriv(xt_bd).view(-1, 1)
        xg_init_deriv_train, xg_init_deriv_val, xg_init_deriv_test = self.split_data(xt_init_deriv, g_init_deriv, max(10, math.floor(p_train * n_tot)), max(10, math.floor(p_val * n_val)))

        data_train = ((x_int_train, f_train), xg_train, xg_init_train, xg_init_deriv_train)
        data_val = ((x_int_val, f_val), xg_val, xg_init_val, xg_init_deriv_val)
        data_test = ((x_int_test, f_test), xg_test, xg_init_test, xg_init_deriv_test)
        return data_train, data_val, data_test


def pde_libraryWaveEquation1D(fctn_num=0):

    if fctn_num == 0:
        c = 2.0
        p = torch.pi
        u_true = lambda xt: torch.sin(p * xt[:, 0] + torch.cos(p * xt[:, 0])) * torch.exp(-xt[:, 1])
        f = lambda xt: 0 * xt[:, 0]
        g_init = lambda xt: 0

        # u_true = lambda xt: x_part(xt[:, 0]) * t_part(xt[:, 1])  # true solution (unknown in practice)
        # f_true = lambda xt: x_part(xt[:, 0]) * dt_part(xt[:, 1]) - dx2_part(xt[:, 0]) * t_part(xt[:, 1])  # source term (in this case, f = du / dt - d^2u/dx^2)
        # g_true = lambda xt: x_part(xt[:, 0]) * t_part(t_domain[0])  # initial condition
        # b_true_0 = lambda xt: x_part(x_domain[0]) * t_part(xt[:, 1])  # Dirichlet boundary condition
        # b_true_1 = lambda xt: x_part(x_domain[1]) * t_part(xt[:, 1])  # Dirichlet boundary condition
        pde = None
        pde = {'eqn': 'd2u/dt2 - c^2 * d^2u/dx^2 = f; u_{bd} = g; u_{init} = g_init',
               'u_true': u_true,
               'f': lambda xt: (c ** 2 - p ** 2) * u_true(xt),
               'g': u_true,
               'g_init': lambda xt: torch.sin(p * xt[:, 0] + torch.cos(p * xt[:, 0])),
               'g_init_deriv': lambda xt: -(torch.sin(p * xt[:, 0] + torch.cos(p * xt[:, 0])))}
    else:
        raise ValueError('fctn_num = 0 for Heat Equation1D')

    return pde


if __name__ == "__main__":
    import torch.nn as nn
    # from dnn101.pinns.pde_data import plot_data

    pde_setup = pde_libraryWaveEquation1D(0)
    f_true = pde_setup['f']
    g_true = pde_setup['g']
    g_init = pde_setup['g_init']
    g_init_deriv = pde_setup['g_init_deriv']
    u_true = pde_setup['u_true']

    pde = DNN101DataPINNWaveEquation1D(f_true, g_true, g_init, g_init_deriv, domain=PDEDomainBox((-1, 1, 0, 10)), u_true=u_true)

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


    for t in range(0, 10, 2):
        pde.plot_slice(net2D, t, label=t)

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
    # pde.plot_slice(0)
    # plt.show()

