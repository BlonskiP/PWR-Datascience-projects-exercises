from __future__ import annotations

import abc
import math
import typing as t
import matplotlib.pyplot as plt
import numpy as np
import pyro
import pyro.distributions as dist
import torch
import random
from matplotlib.patches import Ellipse
from pyro import poutine
from pyro.contrib.autoguide import AutoDelta
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate
from torch.distributions import constraints
import matplotlib.pyplot as plt
import torch


class MixtureModel(abc.ABC):
    def __init__(self, num_components: int):
        self.num_components = num_components
        self.history = {
            "likelihood": []
        }  # TODO: accumulate during training

    @abc.abstractmethod
    def fit(self, x: torch.Tensor) -> MixtureModel:
        """Fit posterior of k Gaussians and mixing components.

        :param x: N x D matrix of the input data points.
        :return: self
        """

    @abc.abstractmethod
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predict assignment probabilities for the input points.

        :param x: N x D matrix of the input data points.
        :return: N x K matrix of probabilities where K is the number of mixing
            components of gaussians.
        """

    @abc.abstractmethod
    def log_likelihood(self, x: torch.Tensor) -> t.Union[float, torch.Tensor]:
        """Calculate log likelihood of determined posterior.

        :param x: N x D matrix of the input data points.
        :return: A scalar containing log likelihood.
        """

    @abc.abstractmethod
    def score(self, x: torch.Tensor):
        """Calculate negative log likelihood for each sample and component.

        :param x: N x D matrix of the input data points.
        :return: N x K matrix containing negative log likelihood of a sample
            being produced by each of the K components in the mixture model.
        """


class GaussianMixtureModel(MixtureModel):
    def __init__(self, num_components: int, optim_steps: int):
        super().__init__(num_components)
        self.optim_steps = optim_steps
        self.delta = 1e-8
        self.eps = 1.e-6
        self.num_components = num_components
        self.DIM = 2
        self._log_likelihood = -np.inf
        self.mu = torch.nn.Parameter(torch.randn(1, self.num_components, self.DIM), requires_grad=False) # means parameter for multiplie gausians.
        self.var = torch.nn.Parameter(torch.ones(1, self.num_components, self.DIM), requires_grad=False) #variance parameter for gaucians SIGMA
        self.pi = torch.nn.Parameter(torch.Tensor(1, self.num_components, 1), requires_grad=False)\
            .fill_(1. / self.num_components)  # probablity PI

    def __p_k(self, x, mu, var):
        mu = mu.expand(x.size(0), self.num_components, self.DIM)
        var = var.expand(x.size(0), self.num_components, self.DIM)

        exponent = torch.exp(-.5 * torch.sum((x - mu) * (x - mu) / var, 2, keepdim=True))
        prefactor = torch.rsqrt(((2. * math.pi) ** self.DIM) * torch.prod(var, dim=2, keepdim=True) + self.eps)

        return prefactor * exponent

    def __e_step(self, pi, p_k):#Exceptation
        weights = pi * p_k
        return torch.div(weights, torch.sum(weights, 1, keepdim=True) + self.eps)

    def __m_step(self, x, weights):#Maximalizaion
        n_k = torch.sum(weights, 0, keepdim=True) #mc


        pi_new = torch.div(n_k, torch.sum(n_k, 1, keepdim=True) + self.eps)#pi=mc/m
        mu_new = torch.div(torch.sum(weights * x, 0, keepdim=True), n_k + self.eps)
        var_new = torch.div(torch.sum(weights * (x - mu_new) * (x - mu_new), 0, keepdim=True), n_k + self.eps)

        return pi_new, mu_new, var_new

    def __em_step(self, x):
        weights = self.__e_step(self.pi, self.__p_k(x, self.mu, self.var))

        pi_new, mu_new, var_new = self.__m_step(x, weights)

        self.__update_pi(pi_new)
        self.__update_mu(mu_new)
        self.__update_var(var_new)


    def __update_mu(self, mu):
        if mu.size() == (self.num_components, self.DIM):
            self.mu = mu.unsqueeze(0)
        elif mu.size() == (1, self.num_components, self.DIM):
            self.mu.data = mu

    def __update_var(self, var):
        if var.size() == (self.num_components, self.DIM):
            self.var = var.unsqueeze(0)
        elif var.size() == (1, self.num_components, self.DIM):
            self.var.data = var

    def __update_pi(self, pi):
        self.pi.data = pi


    def fit(self, x: torch.Tensor) -> MixtureModel:
        """Fit posterior of k Gaussians and mixing components.

        :param x: N x D matrix of the input data points.
        :return: self
        """
        if len(x.size()) == 2:
            # (n, d) --> (n, k, d)
            x = x.unsqueeze(1).expand(x.size(0), self.num_components, x.size(1))

        i = 0
        j = np.inf

        while (i <= self.optim_steps) and (j >= self.delta):

            log_likelihood_old = self._log_likelihood
            mu_old = self.mu
            var_old = self.var

            self.__em_step(x)
            self._log_likelihood = self.log_likelihood(x)

            self.history["likelihood"].append(self._log_likelihood)

            if (self._log_likelihood.abs() == float("Inf")) or (self._log_likelihood == float("nan")):
                self.__init__(self.num_components, self.DIM)

            i += 1
            j = self._log_likelihood - log_likelihood_old

            if j <= self.delta:
                self.__update_mu(mu_old)
                self.__update_var(var_old)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predict assignment probabilities for the input points.

        :param x: N x D matrix of the input data points.
        :return: N x K matrix of probabilities where K is the number of mixing
            components of gaussians.
        """
        if len(x.size()) == 2:
            x = x.unsqueeze(1).expand(x.size(0), self.num_components, x.size(1))

        p_k = self.__p_k(x, self.mu, self.var)

        proba = torch.squeeze(p_k / (p_k.sum(1, keepdim=True) + self.eps), dim=-1)

        return proba


    def log_likelihood(self, x: torch.Tensor) -> t.Union[float, torch.Tensor]:
        """Calculate log likelihood of determined posterior.

        :param x: N x D matrix of the input data points.
        :return: A scalar containing log likelihood.
        """

        if len(x.size()) == 2:
            # (n, d) --> (n, k, d)
            x = x.unsqueeze(1).expand(x.size(0), self.num_components, x.size(1))

        weights = self.pi * self.__p_k(x, self.mu, self.var)

        return torch.sum(torch.log(torch.sum(weights, 1) + self.eps))


    def score(self, x: torch.Tensor):
        """Calculate negative log likelihood for each sample and component.

        :param x: N x D matrix of the input data points.
        :return: N x K matrix containing negative log likelihood of a sample
            being produced by each of the K components in the mixture model.
        """
        if len(x.size()) == 2:
            # (n, d) --> (n, k, d)
            x = x.unsqueeze(1).expand(x.size(0), self.num_components, x.size(1))

        #proba = self.pi * self.__p_k(x, self.mu, self.var)
        #looks better
        proba = self.predict_proba(x)
        score = -torch.log(proba + self.eps)
        #print(score.size())
        #print(self.mu)
        return score
    