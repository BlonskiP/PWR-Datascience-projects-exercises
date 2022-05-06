"""Implementation of Maximum A Posteriori."""
from typing import Dict, List, Union

import torch
from tqdm.auto import tqdm

from src.mm import vis


def run_map(
    prior_model: torch.nn.Module,
    likelihood_model: torch.nn.Module,
    data: List[Union[int, List[int]]],
    num_epochs: int,
    lr: float,
    verbose=False,
) -> Dict[str, Union[List[float], Dict[str, List[float]]]]:
    """Implements Maximum Likelihood Estimation using gradient descent."""
    data = torch.tensor(data, dtype=torch.float)
    history = {
        'losses': [],
        'param_values': [],
    }

    optim = torch.optim.Adam(likelihood_model.parameters(), lr=lr)

    for _ in tqdm(range(num_epochs), disable=not verbose):
        # Zero gradients
        optim.zero_grad()

        # Compute predictions
        prior = prior_model(likelihood_model.probs)
        likelihood = likelihood_model(data)
    
        # Compute loss
        loss = -1 * (torch.sum(prior) + torch.sum(likelihood)) #sum of logarithms

        # Backpropagate
        loss.backward()
        optim.step()

        # Save values to log
        history['losses'].append(loss.item())

        mp = likelihood_model.probs
        history['param_values'].append(
            mp.item() if len(mp) == 1 else mp.tolist()
        )

    if verbose:
        vis.plot_losses_over_time(history['losses'])

    return history