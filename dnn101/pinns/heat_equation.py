
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
from dnn101.pinns import PDEDomain, PDEDomainBox



class HeatEquation1DPINN(torch.nn.Module):

    def __init__(self, net, x_domain, t_domain):
        super(HeatEquation1DPINN, self).__init__()
        self.net = net

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

        return u_t - u_xx


class DNN101DataPINNHeatEquation1D(DNN101Data):
    def __init__(self, f: Callable, g: Callable, g_init: Callable, domain: PDEDomain = PDEDomainBox((-1, 1, -1, 1)), u_true: Callable = None):
        super(DNN101DataPINNHeatEquation1D, self).__init__()
        self.f = f
        self.g = g
        self.g_init = g_init
        self.domain = domain # [x_low, x_high, t_low, t_high]
        self.u_true = u_true

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
        xt_init = xt_bd[:xt_bd.shape[0] // 4]
        xt_bd = xt_bd[xt_bd.shape[0] // 2:]
        g = self.g(xt_bd).view(-1, 1)

        n_tot = xt_bd.shape[0]
        p_train = n_train / n_int
        p_val = n_val / n_int

        (x_bd_train, g_train), (x_bd_val, g_val), (x_bd_test, g_test) = self.split_data(xt_bd, g, max(10, math.floor(p_train * n_tot)), max(10, math.floor(p_val * n_val)))


        # true initial condition
        g_init = self.g_init(xt_init)

        n_tot = xt_init.shape[0]
        p_train = n_train / n_int
        p_val = n_val / n_int

        (x_init_train, g_init_train), (x_init_val, g_init_val), (x_init_test, g_init_test) = self.split_data(xt_init, g_init, max(10, math.floor(p_train * n_tot)), max(10, math.floor(p_val * n_val)))

        data_train = ((x_int_train, f_train), (x_bd_train, g_train), (x_init_train, g_init_train))
        data_val = ((x_int_val, f_val), (x_bd_val, g_val), (x_init_val, g_init_val))
        data_test = ((x_int_test, f_test), (x_bd_test, g_test), (x_init_test, g_init_test))
        return data_train, data_val, data_test

    def plot_data(self, *args, label='train', marker='o'):
        # data x_int, y_int, x_bd, y_bd
        color_order = ['b', 'r', 'g']
        label_order = [': int', ': bd', ': init']
        for count, i in enumerate(range(0, len(args), 2)):
            x = args[i]
            y = args[i + 1]
            if x is not None and y is not None:
                plt.scatter(x[:, 0], x[:, 1], color=color_order[count], marker=marker, label=label + label_order[count])

        plt.xlabel('x')
        plt.ylabel('t')
        plt.legend()

    def plot_prediction(self, net, *args):
        with torch.no_grad():
            x1_grid, x2_grid = self.domain.generate_grid2D()
            x_grid = torch.cat((x1_grid.reshape(-1, 1), x2_grid.reshape(-1, 1)), dim=1)
            plt.contourf(x1_grid, x2_grid, net(x_grid).reshape(x1_grid.shape).detach())

    def plot_prediction2(self, net, *args):
        with torch.no_grad():
            x1_grid, x2_grid = self.domain.generate_grid2D()
            x_grid = torch.cat((x1_grid.reshape(-1, 1), x2_grid.reshape(-1, 1)), dim=1)

            if self.u_true is not None:
                plt.subplot(1, 3, 1)
                plt.contourf(x1_grid, x2_grid, self.u_true(x_grid).reshape(x1_grid.shape))
                plt.xlabel('x')
                plt.ylabel('t')
                plt.colorbar()
                plt.title('true')

                plt.subplot(1, 3, 2)
                plt.contourf(x1_grid, x2_grid, net(x_grid).reshape(x1_grid.shape))
                plt.xlabel('x')
                plt.ylabel('t')
                plt.colorbar()
                plt.title('approx')

                plt.subplot(1, 3, 3)
                plt.contourf(x1_grid, x2_grid, torch.abs(net(x_grid).reshape(-1) - self.u_true(x_grid).reshape(-1)).reshape(x1_grid.shape))
                plt.xlabel('x')
                plt.ylabel('t')
                plt.colorbar()
                plt.title('abs. diff.')
            else:
                plt.contourf(x1_grid, x2_grid, net(x_grid).reshape(x1_grid.shape))
                plt.xlabel('x')
                plt.ylabel('t')
                plt.colorbar()
                plt.title('approx')


    def plot_slice(self, t, net=None, show_diff=False):
        x1_grid = torch.linspace(self.domain.domain[0], self.domain.domain[1], 50)
        x_grid = torch.cat((x1_grid.reshape(-1, 1), t * torch.ones_like(x1_grid).reshape(-1, 1)), dim=1)

        if net is None:
            if u_true is not None:
                u = self.u_true(x_grid)
                label = 'true'
            else:
                print('no slice to plot!')
        else:
            u = net(x_grid)
            label = 'approx'
            if show_diff is True and self.u_true is not None:
                u = u - self.u_true(x_grid)
                label = 'abs. diff.'

        plt.plot(x1_grid.view(-1), u.view(-1).detach(), label=label)
        plt.xlim(self.domain.domain[0], self.domain.domain[1])



def pde_libraryHeatEquation1D(fctn_num=0):

    if fctn_num == 0:
        x_part = lambda x: torch.exp(-x ** 2 + 1) - 1
        dx_part = lambda x: -2 * x * torch.exp(-x ** 2 + 1)
        dx2_part = lambda x: (-2 + 4 * x) * torch.exp(-x ** 2 + 1)
        t_part = lambda t: (t + 1) ** (-1 / 2)
        dt_part = lambda t: -(t + 1) ** (-3 / 2)

        u_true = lambda xt: x_part(xt[:, 0]) * t_part(xt[:, 1])  # true solution (unknown in practice)
        f_true = lambda xt: x_part(xt[:, 0]) * dt_part(xt[:, 1]) - dx2_part(xt[:, 0]) * t_part(xt[:, 1])  # source term (in this case, f = du / dt - d^2u/dx^2)
        # g_true = lambda xt: x_part(xt[:, 0]) * t_part(t_domain[0])  # initial condition
        # b_true_0 = lambda xt: x_part(x_domain[0]) * t_part(xt[:, 1])  # Dirichlet boundary condition
        # b_true_1 = lambda xt: x_part(x_domain[1]) * t_part(xt[:, 1])  # Dirichlet boundary condition

        pde = {'eqn': 'du/dt - d^2u/dx^2 = f; u_{bd} = g; u_{init} = g_init',
               'u_true': u_true,
               'f': f_true,
               'g': u_true,
               'g_init': u_true}
    else:
        raise ValueError('fctn_num = 0 for Heat Equation1D')

    return pde


if __name__ == "__main__":
    import torch.nn as nn

    pde_setup = pde_libraryHeatEquation1D(0)
    f_true = pde_setup['f']
    g_true = pde_setup['g']
    g_init = pde_setup['g_init']
    u_true = pde_setup['u_true']

    pde = DNN101DataPINNHeatEquation1D(f_true, g_true, g_init, domain=PDEDomainBox((-1, 1, 0, 10)), u_true=u_true)
    (train_int, train_bd, train_init), (val_int, val_bd, val_init), (test_int, test_bd, test_init) = pde.generate_data()
    pde.plot_data(*train_int, *train_bd, *train_init, marker='o', label='train')
    pde.plot_data(*val_int, *val_bd, *val_init, marker='s', label='val')
    pde.plot_data(*test_int, *test_bd, *test_init, marker='^', label='test')
    plt.show()

    net2D = nn.Sequential(nn.Linear(2, 10),
                          nn.Tanh(),
                          nn.Linear(10, 1))
    pde.plot_prediction2(net2D)
    plt.show()

    pde.plot_slice(pde.domain.domain[2])
    plt.ylim([0, 2])
    plt.show()
