import torch
from copy import deepcopy


class PDEDomain:
    def __init__(self, domain):
        self.domain = domain

    def generate_interior_points(self, n):
        raise NotImplementedError

    def generate_boundary_points(self, n):
        raise NotImplementedError

    def generate_grid2D(self, n=50):
        raise NotImplementedError


class PDEDomainBox(PDEDomain):

    def __init__(self, domain: tuple = (-1, 1, -1, 1)):
        super(PDEDomainBox, self).__init__(domain)
        self.d = len(domain)

    def generate_interior_points(self, n):
        pts_int = torch.empty(0)
        for i in range(0, self.d, 2):
            x = self.domain[i] + (self.domain[i + 1] - self.domain[i]) * torch.rand(n)
            pts_int = torch.cat((pts_int, x.reshape(-1, 1)), dim=1)
        return pts_int

    def generate_boundary_points(self, n):
        # generate points on boundary
        left_boundary = []
        right_boundary = []
        for i in range(0, self.d, 2):
            left_boundary.append(self.domain[i] * torch.ones(n, 1))
            right_boundary.append(self.domain[i + 1] * torch.ones(n, 1))

        pts_bd = torch.empty(0)
        for count, i in enumerate(range(0, len(self.domain), 2)):
            x = self.domain[i] + (self.domain[i + 1] - self.domain[i]) * torch.rand(n).reshape(-1, 1)

            left_x = deepcopy(left_boundary)
            left_x[count] = x

            x = self.domain[i] + (self.domain[i + 1] - self.domain[i]) * torch.rand(n).reshape(-1, 1)
            right_x = deepcopy(right_boundary)
            right_x[count] = x

            pts_bd = torch.cat((pts_bd, torch.cat(tuple(left_x), dim=1), torch.cat(tuple(right_x), dim=1)), dim=0)

        return pts_bd

    def generate_grid2D(self, n=50):
        x1_grid, x2_grid = torch.meshgrid(torch.linspace(self.domain[0], self.domain[1], n),
                                          torch.linspace(self.domain[2], self.domain[3], n),
                                          indexing='ij')

        return x1_grid, x2_grid


class PDEDomainEllipse(PDEDomain):
    def __init__(self, center: tuple = (0, 0), radii: tuple = (1, 1)):
        super(PDEDomainEllipse, self).__init__((center[0] - radii[0], center[0] + radii[0], center[1] - radii[1], center[1] + radii[1]))
        self.d = len(center)
        self.center = center
        self.radii = radii

    def generate_interior_points(self, n):
        theta = 2 * torch.pi * torch.rand(n)

        r_x = self.radii[0] * torch.rand(n)
        x = self.center[0] + r_x * torch.cos(theta)

        r_y = self.radii[1] * torch.rand(n)
        y = self.center[1] + r_y * torch.sin(theta)

        return torch.cat((x.reshape(-1, 1), y.reshape(-1, 1)), dim=1)

    def generate_boundary_points(self, n):
        theta = 2 * torch.pi * torch.rand(n)
        x = self.center[0] + self.radii[0] * torch.cos(theta)
        y = self.center[1] + self.radii[1] * torch.sin(theta)

        return torch.cat((x.reshape(-1, 1), y.reshape(-1, 1)), dim=1)

    def generate_grid2D(self, n=50):
        r_grid, theta_grid = torch.meshgrid(torch.linspace(0, self.radii[0], n),
                                            torch.linspace(0, 2 * torch.pi, n),
                                            indexing='ij')

        x1_grid = self.center[0] + r_grid * torch.cos(theta_grid)

        r_grid, theta_grid = torch.meshgrid(torch.linspace(0, self.radii[1], n),
                                            torch.linspace(0, 2 * torch.pi, n),
                                            indexing='ij')

        x2_grid = self.center[1] + r_grid * torch.sin(theta_grid)

        return x1_grid, x2_grid
