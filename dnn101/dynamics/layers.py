import torch
import torch.nn as nn
import math
from copy import deepcopy


class Identity(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input


class AntiSymmetricLayer(nn.Module):

    def __init__(self, in_features, bias=True, gamma=1e-4, activation=None, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(AntiSymmetricLayer, self).__init__()
        self.in_features = in_features
        self.gamma = gamma
        self.activation = activation
        self.weight = nn.Parameter(torch.empty((in_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(in_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        x = nn.functional.linear(x, self.weight - self.weight.t()
                                 - self.gamma * torch.eye(self.in_features, dtype=x.dtype, device=x.device), self.bias)

        if self.activation is not None:
            x = self.activation(x)

        return x

    def extra_repr(self) -> str:
        return 'in_features={}, bias={}'.format(
            self.in_features, self.bias is not None
        )


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


class TruncationLayer(nn.Module):

    def __init__(self, out_features):
        super(TruncationLayer, self).__init__()
        self.out_features = out_features

    def forward(self, xz):
        return xz[:, :self.out_features]


if __name__ == "__main__":

    width = 4
    layer = nn.Linear(3, width)
    hamiltonian_layer = HamiltonianLayer(layer)
    x = torch.randn(11, width)
    y = hamiltonian_layer(x)
    print(y.shape)