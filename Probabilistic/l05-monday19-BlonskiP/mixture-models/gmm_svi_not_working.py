from __future__ import annotations

import abc
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
            "loss": []
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
        self.DIM = 2 #Data dimensions

    def model(self,data):
        weights = pyro.param('weights', torch.FloatTensor([1 / self.num_components for i in range(self.num_components)]),
                             constraint=constraints.unit_interval)
        scales = pyro.param('scales', torch.tensor([np.diag(np.random.choice([1, 1.5, 2, 2.5], self.DIM)) for _ in range(self.num_components)],
                                                   dtype=torch.float), constraint=constraints.positive)  # chyba ok

        locs = pyro.param('locs',
                          torch.tensor([[random.choice([1., 1.5, 1.75, 2]) for i in range(self.DIM)] for j in range(self.num_components)]))

        with pyro.plate('data', data.size(0)):
            assignment = pyro.sample('assignment', dist.Categorical(weights)).to(torch.int64)
            sample = pyro.sample('obs', dist.MultivariateNormal(locs[assignment], scales[assignment]), obs=data)

    @config_enumerate(default="parallel")
    @poutine.broadcast
    def full_guide(self,data):
        with poutine.block(hide_types=["param"]):  # Keep our learned values of global parameters.
            self.global_guide(data)
        with pyro.plate('data', len(data)):
            assignment_probs = pyro.param('assignment_probs', torch.ones(len(data), self.num_components) / self.num_components,
                                      constraint=constraints.unit_interval)
            pyro.sample('assignment', dist.Categorical(assignment_probs))

    def fit(self, data: torch.Tensor) -> MixtureModel:
        K = self.num_components
        pyro.clear_param_store()
        self.global_guide = AutoDelta(poutine.block(self.model, expose=['weights', 'locs', 'scales']))
        self.global_guide = config_enumerate(self.global_guide, 'parallel')
        optim = pyro.optim.Adam({'lr': 0.1, 'betas': [0.8, 0.99]})
        elbo = TraceEnum_ELBO(max_iarange_nesting=1)
        svi = SVI(self.model, self.full_guide, optim, loss=elbo)
        # Initialize weights to uniform.
        pyro.param('auto_weights',
                             torch.FloatTensor([1 / self.num_components for i in range(self.num_components)]),
                             constraint=constraints.unit_interval)
        # Assume half of the data variance is due to intra-component noise.
        pyro.param('auto_scales', torch.tensor(
            [np.diag(np.random.choice([0., 1., 1.5, 2.], self.DIM)) for _ in range(self.num_components)],
            dtype=torch.float), constraint=constraints.positive)
        # Initialize means from a subsample of data.
        pyro.param('auto_locs',
                          torch.tensor([[random.choice([0., 1., 1.5, 2.]) for i in range(self.DIM)] for j in
                                        range(self.num_components)]))

        loss = svi.loss(self.model, self.full_guide, data)
        for step in range(self.optim_steps):
            loss = svi.step(data)
            self.history["loss"].append(loss)

        return self

    def predict_proba(self, data: torch.Tensor) -> torch.Tensor:
        with poutine.block(hide_types=["param"]):  # Keep our learned values of global parameters.
            self.global_guide(data)
        scales = pyro.param('auto_scales', constraint=constraints.positive)
        # Initialize means from a subsample of data.
        locs = pyro.param('auto_locs',
                   torch.tensor([[random.choice([0., 1., 1.5, 2.]) for i in range(self.DIM)] for j in
                                 range(self.num_components)]))
        probs = torch.empty(len(data),self.num_components)
        for K in range(self.num_components):
            distribution = dist.MultivariateNormal(locs[K], scales[K])
            prob = 10**distribution.log_prob(data) #zrobić EXP a nie 10**
            probs[:,K] = prob
        return probs


    def log_likelihood(self, x: torch.Tensor) -> t.Union[float, torch.Tensor]:
        assignment_probs = self.predict_proba(x)
        sum = torch.sum(assignment_probs,dim=1)
        logs = torch.log(sum)
        log_Likelihood = torch.sum(logs)
        return log_Likelihood

    def score(self, x: torch.Tensor):
        return -1*torch.log(self.predict_proba(x))