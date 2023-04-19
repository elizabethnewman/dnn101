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


class TruncationLayer(nn.Module):

    def __init__(self, out_features):
        super(TruncationLayer, self).__init__()
        self.out_features = out_features

    def forward(self, xz):
        return xz[:, :self.out_features]


if __name__ == "__main__":

    layer = AntiSymmetricLayer(3)
    x = torch.randn(11, 3)
    y = layer(x)
    print(y.shape)