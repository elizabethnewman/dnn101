import torch
import torch.nn as nn
import math
from copy import deepcopy


class ResNetLayerRK1(nn.Module):

    def __init__(self, layer: nn.Module, activation: nn.Module = nn.Tanh(), h: float = 1.0):
        super(ResNetLayerRK1, self).__init__()
        self.h = h
        self.layer = layer  # TODO: check in_features == out_features
        self.activation = activation

    def forward(self, x):
        return x + self.h * self.activation(self.layer(x))


class ResNetLayerRK4(nn.Module):
    def __init__(self, layer: nn.Module, activation: nn.Module = nn.Tanh(), h: float = 1.0):
        super(ResNetLayerRK4, self).__init__()
        self.h = h
        # t, t + h / 2, t + h
        self.layer = nn.ModuleList([deepcopy(layer) for _ in range(3)])  # initialize with constant weights
        self.activation = activation

    def forward(self, x):
        # time t
        k1 = self.h * self.activation(self.layer[0](x))

        # time t + h / 2
        k2 = self.h * self.activation(self.layer[1](x + 0.5 * k1))
        k3 = self.h * self.activation(self.layer[1](x + 0.5 * k2))

        # time t + h
        k4 = self.h * self.activation(self.layer[2](x + k3))

        return x + (1.0 / 6.0) * k1 + (1.0 / 3.0) * k2 + (1.0 / 3.0) * k3 + + (1.0 / 6.0) * k4


class HamiltonianLayer(nn.Module):

    def __init__(self, layer: nn.Module, h=1.0, activation: nn.Module = nn.Tanh(), bias=True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(HamiltonianLayer, self).__init__()
        self.h = h
        self.activation = activation

        if not isinstance(layer, nn.Linear):
            raise ValueError('HamiltonianLayer only supports Linear')

        self.in_features = layer.out_features
        self.width = layer.in_features
        self.weight = deepcopy(layer.weight)
        if bias:
            self.bias = nn.Parameter(torch.empty(1, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, xz):

        # update z
        if xz.shape[1] == self.in_features:
            x = xz
            z = torch.zeros(1, device=x.device, dtype=x.dtype)
        else:
            x = xz[:, :self.in_features]
            z = xz[:, self.in_features:]

        dz = nn.functional.linear(x, self.weight.T, self.bias)
        if self.activation is not None:
            dz = self.activation(dz)
        z = z - self.h * dz

        # update x
        dx = nn.functional.linear(z, self.weight, self.bias)
        if self.activation is not None:
            dx = self.activation(dx)
        x = x + self.h * dx

        return torch.cat((x, z), dim=1)

    def extra_repr(self) -> str:
        return 'in_features={}, width={}, bias={}'.format(
            self.in_features, self.width, self.bias is not None
        )


if __name__ == "__main__":
    width = 4
    layer = nn.Linear(width, width)
    activation = nn.Tanh()
    resnet_layer = ResNetLayerRK4(layer, activation=activation, h=0.5)

    x = torch.randn(11, width)
    y = resnet_layer(x)
    print(y.shape)

    conv_layer = nn.Conv2d(3, 3, kernel_size=(3, 3), padding=1)
    activation = nn.Tanh()
    resnet_conv_layer = ResNetLayerRK1(conv_layer, activation=activation, h=0.5)

    width = 4
    layer = nn.Linear(3, width)
    hamiltonian_layer = HamiltonianLayer(layer)
    x = torch.randn(11, width)
    y = hamiltonian_layer(x)
    print(y.shape)





