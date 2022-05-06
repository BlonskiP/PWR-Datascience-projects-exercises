import itertools
from collections import defaultdict
from copy import deepcopy

import numpy as np

from src.hmm import train


def viterbi_decode(theta, X):
    """Implement the Viterbi decoding algorithm.

    :param theta: Parameters of the HMM. Tuple of 3 elements: initial
        state distribution, transition probability, emission probability.
    :param X: Sequence of elements of length T.
    :return: Tuple of: the most probable path of hidden states and the
        omega matrix. The most probability is chosen in a greedy fashion
        by takin the most probable hidden state in each step. Omega describes
        hidden states probabilities at each `t` step.
    """
    # Compute scores (forward pass)
    omega = {}  # (t, z_t) -> max (log_p, z_{t-1})
    max_Z = []  # most probable hidden states
    
    pi, tr, em = theta
    
    # Implementation here
    hidden_states = pi.keys()

    for t in range(len(X)):
        if t == 0:
            max_prob = 0
            max_hidden_state = ''
            for hidden_state, init_prob in pi.items():
                omega[t, hidden_state] = (init_prob * em[(X[t], hidden_state)], '')
                if omega[t, hidden_state][0] > max_prob:
                    max_prob = omega[t, hidden_state][0]
                    max_hidden_state = hidden_state
            max_Z.append(max_hidden_state)
        else:
            max_prob = 0
            max_hidden_state = ''
            for current_hidden_state in hidden_states:
                prev_hidden_state = max_Z[t-1]
                prob = omega[(t-1, prev_hidden_state)][0] * tr[(prev_hidden_state, current_hidden_state)] * em[(X[t], current_hidden_state)]
                omega[(t, current_hidden_state)] = (prob, prev_hidden_state)
                if prob > max_prob:
                    max_prob = prob
                    max_hidden_state = current_hidden_state
            max_Z.append(max_hidden_state)

    return max_Z, omega


class ViterbiTrainingAlgorithm(train.TrainingAlgorithm):
    
    def __init__(self, all_X, all_Z, eps, max_epochs, verbose, smoothing=1):
        super().__init__(all_X, all_Z, eps, max_epochs, verbose)
        self._s = smoothing
    
    def estimate_parameters(self, theta, X):
        """Estimate parameters of the HMM usng Viterbi algorithm.

        :param theta: Parameters of the HMM. Tuple of 3 elements: initial
            state distribution, transition probability, emission probability.
        :param X: Sequence of elements of length T.
        :return: Tuple of parameters of the HMM: initial state probability pi,
            transition probability matrix a, and emission probabiliy matrix b.
        """
        pi, tr, em = deepcopy(theta)

        pi_counts = defaultdict(lambda: self._s)
        tr_counts = defaultdict(lambda: self._s)
        em_counts = defaultdict(lambda: self._s)

        for x in X:
            max_Z, omega = viterbi_decode((pi, tr, em), x)

            for t in range(len(max_Z)):
                if t == 0:
                    pi_counts[max_Z[t]] += 1
                    em_counts[(x[t], max_Z[t])] += 1
                else:
                    tr_counts[(max_Z[t-1], max_Z[t])] += 1
                    em_counts[(x[t], max_Z[t])] += 1

        pi_counts_sum = sum(pi_counts.values())
        # tr_counts_sum = sum(tr_counts.values())
        tr_counts_sum = defaultdict(lambda: 0)
        for key, value in tr_counts.items():
            tr_counts_sum[key[0]] += value
        # em_counts_sum = sum(em_counts.values())
        em_counts_sum = defaultdict(lambda: 0)
        for key, value in em_counts.items():
            em_counts_sum[key[1]] += value

        for key, value in pi.items():
            pi[key] = pi_counts[key] / pi_counts_sum

        for key, value in tr.items():
            tr[key] = tr_counts[key] / tr_counts_sum[key[0]]

        for key, value in em.items():
            em[key] = em_counts[key] / em_counts_sum[key[1]]


        return pi, tr, em