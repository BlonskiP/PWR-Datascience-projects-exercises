import abc
import typing as t

import matplotlib.pyplot as plt
import torch
import math


class Kernel(abc.ABC):

    def apply(
        self,
        points: torch.Tensor,
        other_points: t.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Transforms input data to obtain covariance matrix.

        Covariance matrix is used in the gaussian distribution over functions.
        :param points: Tensor of shape N x D, where N is the number of
            samples and D is the number of dimensions.
        :param other_points: Tensor of shape N' x D, where N' is the number of
            samples and D is the number of dimensions.
        :return: Matrix of size N x N'. If `other_points` is `None`, then the
            method should produce matrix N x N.
        """

        if other_points == None: other_points = points.clone()
        N = len(points)
        N_prim = len(other_points)

        covariance_matrix = torch.ones([N, N_prim])

        for i in range(N):
            for j in range(N_prim):
                covariance_matrix[i,j] = self.eq(points[i], other_points[j])

        return covariance_matrix

    def visualize(self, num_points: int) -> plt.Axes:
        """Visualize kernel calculation.

        Example results are shown in
        https://mlss2011.comp.nus.edu.sg/uploads/Site/lect1gp.pdf.
        :param num_points: Number of points to generate.
        :return: Plot figure with kernel working
        """
        x = torch.arange(num_points).unsqueeze(-1).float()
        num_samples = x.shape[0]
        results = (
            self.apply(x)
            .reshape((num_samples, num_samples))
            .detach()
            .cpu()
            .numpy()
        )

        fig, ax = plt.subplots(1, 1)
        plt.imshow(results)

        return fig


class RBFKernel(Kernel):
    def __init__(self, variance: float, lengthscale: float):
        self.lengthscale = lengthscale
        self.variance = variance

    def eq(self, x1, x2):
        nominator = -((x1-x2)**2)
        denominator = (2*self.lengthscale**2)
        res = self.variance * math.exp(nominator/denominator)
        return res


class PeriodicKernel(Kernel):
    def __init__(
        self, lengthscale: float, periodicty: float, deviation: float
    ):
        self.lengthscale = lengthscale
        self.periodicity = periodicty
        self.deviation = deviation

    def eq(self, x1, x2):
        nominator = -(2*math.sin(math.pi*abs(x1-x2)/self.periodicity)**2)
        denominator = (self.lengthscale**2)
        res = self.deviation**2 * math.exp(nominator/denominator)
        return res


class KernelCombiner(Kernel):
    def __init__(self, *kernels: Kernel):
        self.kernels = kernels

    def apply(self,points: torch.Tensor,other_points: t.Optional[torch.Tensor] = None,) -> torch.Tensor:
        if other_points == None: other_points = points.clone()

        N = len(points)
        N_prim = len(other_points)
        cov_matrix = torch.ones([N, N_prim])

        for kernel in self.kernels:
            cov_matrix *= kernel.apply(points, other_points)
        return cov_matrix
