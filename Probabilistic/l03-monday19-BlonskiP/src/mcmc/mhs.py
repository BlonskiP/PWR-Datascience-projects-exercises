from collections import defaultdict
from typing import Dict

import matplotlib.pyplot as plt
import pyro
import pyro.distributions as dist
import torch
import numpy as np

from pyro.infer.mcmc import mcmc_kernel
from pyro.infer.mcmc.util import initialize_model


class MetropolisHastings(mcmc_kernel.MCMCKernel):
    """Implementation of Metropolis Hastings sampler for MCMC."""

    def __init__(self, model, proposal_dist, priors):
        """Inits MetropolisHastings.

        :param model: Probabilistic model to estimate (likelihood).
        :param proposal_dist: Distribution to generate next parameter value.
        :param priors: Prior distribution over parameter space.
        """
        self.model = model
        self._proposal_dist = proposal_dist
        self._priors = priors

        self._model_args = None
        self._model_kwargs = None

        self._initial_params = None

        self._step = 0
        self._warmup_steps = None

        self._generated_samples = {
            'accepted': defaultdict(list),
            'rejected': defaultdict(list),
            'counts': {
                'accepted': 0,
                'rejected': 0,
            }
        }

    @property
    def initial_params(self):
        return self._initial_params

    @initial_params.setter
    def initial_params(self, params):
        self._initial_params = params

    def logging(self):
        """Provides statistics for progress bar."""
        return {
            '#accepted': self._generated_samples['counts']['accepted'],
            '#rejected': self._generated_samples['counts']['rejected'],
        }

    def setup(self, warmup_steps, *args, **kwargs):
        """Sets up the sampler."""
        self._warmup_steps = warmup_steps

        init_params, _, _, _ = initialize_model(
            self.model, args, kwargs,
        )
        if self._initial_params is None:
            self._initial_params = init_params

        self._model_args = args
        self._model_kwargs = kwargs

    def _next_parameters_proposal(
        self,
        curr_params: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Samples new parameters from the proposal distribution.

        Use `pyro.sample` and include current step in the sample name,
        eg. `c_0`, `c_1`...

        :param curr_params: Current parameter values.
        :return: New  parameter values.
        """
        # EXCERSISE TO IMPLEMENT
        new_currs = dict()
        #samples C_X example C_0
        new_currs['c'] = pyro.sample(f'c_{self._step}', self._proposal_dist(curr_params['c']))
        # std
        new_currs['std'] = pyro.sample(f'std_{self._step}', self._proposal_dist(curr_params['std']))
        return new_currs


    def _get_log_likelihood(self, params):
        """Calculates the log-likelihood of the provided params.

        Use `pyro.condition` and `pyro.poutine.trace`.
        """
        #http://docs.pyro.ai/en/0.2.1-release/poutine.html
        # EXCERSISE TO IMPLEMENT
        conditioned_model = pyro.poutine.condition(self.model, params)
        traced_model = pyro.poutine.trace(conditioned_model).get_trace(
            *self._model_args, **self._model_kwargs)
        return traced_model.log_prob_sum()

    def _get_log_priors(self, params):
        """Calculates log-prob of prior distribution for the provided params."""
        # EXCERSISE TO IMPLEMENT
        vals = [self._priors[pname](pval) for pname, pval in params.items()]
        return torch.tensor(vals).log().sum()

    def _should_accept(self, curr_lp, new_lp):
        """Decides whether to accept or reject the new params configuration."""
        #EXCERSISE TO IMPLEMENT
        #Generate new candidate
        log_priors_curr = self._get_log_priors(curr_lp)
        log_priors_new = self._get_log_priors(new_lp)
        log_likelihood_curr = self._get_log_likelihood(curr_lp)
        log_likelihood_new = self._get_log_likelihood(new_lp)
        #Check acceptence
        if log_likelihood_new + log_priors_new > log_likelihood_curr + log_priors_curr:
            return True
        else:
            #generate uniform random number u in [0,1]
            u = np.random.uniform(0, 1)
            alpha = np.exp(log_likelihood_new + log_priors_new - log_likelihood_curr + log_priors_curr)
            #Accept if u is lower than alpha
            condition = u < alpha
            return condition

    def sample(self, params):
        """Returns new params configuration given current.

        The man body of the Metropolis Hastings sampling algorithm.
        """
        new_params = self._next_parameters_proposal(params) #new param to check
        accept = self._should_accept(params, new_params)  #accept form acepting function

        if accept:
            params = new_params

        if self._step > self._warmup_steps:
            for pname in params.keys():
                k = 'accepted' if accept else 'rejected'
                self._generated_samples[k][pname].append(
                    (self._step, new_params[pname].item())
                )
                self._generated_samples['counts'][k] += 1

        self._step += 1

        return params.copy()


def plot_accepted_rejected_samples(kernel):
    """Plots samples generated by the `kernel` dung estimation."""
    samples = kernel._generated_samples

    fig, axs = plt.subplots(ncols=2, figsize=(15, 5))

    for name, ax in zip(('c', 'std'), axs.ravel()):
        x_rej, y_rej = zip(*samples['rejected'][name])
        ax.plot(
            x_rej, y_rej,
            marker='o', linestyle='',
            color='red', label='Rejected',
            alpha=0.5
        )

        x_acc, y_acc = zip(*samples['accepted'][name])
        ax.plot(
            x_acc, y_acc,
            marker='X', linestyle='',
            color='green', label='Accepted',
            ms=10
        )

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Value')
        ax.set_title(name)
        ax.legend()

    plt.show()
