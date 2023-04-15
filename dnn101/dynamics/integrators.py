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





