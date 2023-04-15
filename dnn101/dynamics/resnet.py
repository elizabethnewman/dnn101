import torch
import torch.nn as nn
from dnn101.dynamics import ResNetLayerRK1, ResNetLayerRK4, HamiltonianLayer, TruncationLayer


class ResNet(nn.Module):

    def __init__(self, layer: nn.Module, activation: nn.Module = nn.Tanh(), depth: int = 2, h: float = 1.0, integrator: str = 'RK1'):
        super(ResNet, self).__init__()

        if integrator == 'RK1':
            layers = nn.ModuleList([ResNetLayerRK1(layer, activation=activation, h=h) for _ in range(depth)])
        elif integrator == 'RK4':
            layers = nn.ModuleList([ResNetLayerRK4(layer, activation=activation, h=h) for _ in range(depth)])
        elif integrator == 'Hamiltonian':
            layers = nn.ModuleList([HamiltonianLayer(layer, activation=activation, h=h) for _ in range(depth)])
            layers.append(TruncationLayer(layer.in_features))
        else:
            raise ValueError('only RK1 and RK4 supported')
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    width = 4
    layer = nn.Linear(width, width)
    activation = nn.Tanh()

    resnet = ResNet(layer, activation=activation, h=0.5, depth=5, integrator='Hamiltonian')

    x = torch.randn(11, width)
    y = resnet(x)
    print(y.shape)

