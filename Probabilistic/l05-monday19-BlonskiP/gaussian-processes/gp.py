from __future__ import annotations

import abc
import typing as t

import matplotlib.pyplot as plt
import torch
import torch.distributions as dist
import numpy as np
import scipy.stats as st
from kernels import Kernel


class Regressor(abc.ABC):
    @abc.abstractmethod
    def fit_train(self, x: torch.Tensor, y: torch.Tensor) -> Regressor:
        """Fit to the training data.

        :param x: N x D matrix of the input data points.
        :param y: N x K matrix of targets for K predicted values.
        :return: self
        """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> t.Any:
        """Predict value for each point in the input data.

        It also should save parameters of the multivariate distribution for
        given points so it can be used `sample_fun` method.

        :param x: N x D matrix of the input data points.
        :return: Anything that you need.
        """

    @abc.abstractmethod
    def confidence_interval(self, x: torch.Tensor) -> torch.Tensor:
        """For each sampling point return its upper and lower confidences.

        It should be realised .95 confidence interval.
        :param x: Input data points of N' x D dimensions.
        :return: Confidence intervals of N' x 2 (lower and upper bounds).
        """

    @abc.abstractmethod
    def log_likelihood(self) -> t.Union[float, torch.Tensor]:
        """Calculate log probability of the fitted model.

        :return: scalar with the log likelihood value.
        """

    @abc.abstractmethod
    def sample_fun(self):
        """Samples single function using test data.

        Test data should be used in `forward` method before calling this
        method.
        :return: M x 1 function values where M is number of data points used
            in the `forward` method.
        """


class GPRegressor(Regressor):
    def __init__(self, kernel: Kernel, noise: float, jitter: float):
        self.kernel = kernel
        self.noise = noise
        self.jitter = jitter # eng. (drżenie) do robienia szumu gdy nie da sie inverse matrix zrobić
                             # inverse_cpu: U(10,10) is zero, singular U. itp

    def fit_train(self, x: torch.Tensor, y: torch.Tensor) -> Regressor:
        noise_matrix = torch.eye(len(x)) * self.noise
        self.cov_matrix = self.kernel.apply(x) + noise_matrix #covariance matrix + noice
        self.x = x
        self.y = y

    def forward(self, x: torch.Tensor) -> t.Any: # Predict value for each point in the input data.
        self.ker_1_t = self.kernel.apply(self.x, x).t()
        self.ker_2 = self.kernel.apply(x)
        self.mi = torch.mm(self.ker_1_t.t(), self.cov_matrix.inverse())
        self.mi = self.mi @ self.y
        self.sigma = self.ker_2 - (self.ker_1_t.t() @ self.cov_matrix.inverse() @ self.ker_1_t)

    def confidence_interval(self, x: torch.Tensor) -> torch.Tensor: # For each sampling point return its upper and lower confidences.
        self.forward(x)
        samples_number = 100
        N = len(x)
        samples = torch.ones(samples_number, N)
        res = torch.ones(N, 2)

        for i in range(samples_number):
            samples[i] = self.sample_fun()
        for i in range(N):
            c = st.norm.interval(0.95, loc=samples[:,i].mean(), scale=samples[:,i].var())
            res[i,0] = c[0]
            res[i,1] = c[1]
        return res #Confidence intervals of N' x 2 (lower and upper bounds).

    def log_likelihood(self) -> t.Union[float, torch.Tensor]:
        mu = self.y
        sigma = self.cov_matrix
        res = dist.MultivariateNormal(mu, sigma).log_prob(self.y)
        return res #scalar with the log likelihood value.

    def sample_fun(self):
        #example_matrix += torch.eye(len(example_matrix)) * 1e-6
        noice = torch.eye(self.sigma.size()[0])*self.jitter
        sigma = self.sigma + 2*noice
        return dist.MultivariateNormal(self.mi, sigma).sample(torch.tensor([1])).squeeze()



def visualize_data_and_intervals(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    data_points_to_visualize: torch.Tensor,
    model: GPRegressor,
) -> plt.Axes:
    intervals = model.confidence_interval(data_points_to_visualize)
    mean = intervals.mean(dim=1)
    fig, ax = plt.subplots(1, 1, figsize=(18, 6))
    ax.plot(data_points_to_visualize.squeeze(), mean, color="blue", lw=2)
    ax.scatter(x_train, y_train.squeeze(), color="green", label="Train")
    ax.scatter(x_test, y_test.squeeze(), color="orange", label="Test")
    ax.fill_between(
        data_points_to_visualize.squeeze(),
        intervals[:, 0],
        intervals[:, 1],
        alpha=0.3,
    )
    ax.legend()
    return ax


def visualize_data_samplings(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    data_points_to_visualize: torch.Tensor,
    model: GPRegressor,
    num_iterations: int,
) -> plt.Axes:
    fig, ax = plt.subplots(1, 1, figsize=(18, 6))
    model.forward(data_points_to_visualize)
    for _ in range(num_iterations):
        ax.plot(
            data_points_to_visualize, model.sample_fun(), color="blue", lw=2
        )

    intervals = model.confidence_interval(data_points_to_visualize)
    ax.scatter(x_train, y_train, color="green", label="Train")
    ax.scatter(x_test, y_test, color="orange", label="Test")
    ax.fill_between(
        data_points_to_visualize.squeeze(),
        intervals[:, 0],
        intervals[:, 1],
        alpha=0.3,
    )
    ax.legend()
    return ax
