import numpy as np
import math

from src.hmm import utils


def forward(theta, X):
    """Implement the forward algorithm.

    :param theta: Parameters of the HMM. Tuple of 3 elements: initial
        state distribution, transition probability, emission probability.
    :param X: Sequence of elements of length T.
    :return: alpha. Dict of log-probabilities, where each key (t, z) denotes
        that any particular state z is chosen at the step t.
    """
    alpha = {}  # (t, z_j) -> log_p

    pi, tr, em = theta
    # Implementation here
    # TOPIC IN JUPYTER NOTEBOOK : Probability of observing given sequence - Matrix of all transition probabiliteis - Alpha
    h_states = pi.keys()
    # T from equasion is called observation. "compute its probabilities in all possible hidden states" etc
    for observation in range(len(X)):
        if observation > 1:
            for current_hidden_state in h_states:
                prob = 0
                for prev_hidden_state in h_states:
                    prob += alpha[(observation - 1, prev_hidden_state)] * tr[
                        (prev_hidden_state, current_hidden_state)] * em[(X[observation], current_hidden_state)]
                alpha[(observation, current_hidden_state)] = prob
        else:  # alpha if t > 1 - > PIj * bjk
            for hidden_state, init_prob in pi.items():
                alpha[observation, hidden_state] = init_prob * em[(X[observation], hidden_state)]
    return alpha


def score_observation_sequence(theta, X):
    """Computes the probability of observing a given sequence.

    :param theta: Parameters of the HMM. Tuple of 3 elements: initial
        state distribution, transition probability, emission probability.
    :param X: Sequence of elements of length T.
    :return: Tuple of log probability and alpha matrix calculated using
        `forward` step. Log probabilities are estimated as a sum of log
        probs over each possible state in the last step of alpha. Note, that
        these probabilities does not have to sum to one, since they
        describe probability of all paths leading to a prticular state.
    """

    pi, tr, em = theta
    # TOPIC IN JUPYTER NOTEBOOK : Probability of observing given sequence - Compute the probablity of observing the given sequence X
    alpha_forward = forward(theta, X)  # alpha from forward.
    t = len(X) - 1  # last obs
    prob = 0
    hidden_states = pi.keys()

    for hs in hidden_states:
        prob += alpha_forward[(t, hs)]
    result = (math.log(prob), alpha_forward)  # Tuple of log probability and alpha matrix calculated using
    return result


def backward(theta, X):
    """Implement the backward algorithm.

    Similarily to `forward`, it estimates probabilities of a state being
    chosen at the step `t`, but these probabilities are conditioned on
    states in a sequence {T, T-1, ..., 1, 0}.
    :param theta: Parameters of the HMM. Tuple of 3 elements: initial
        state distribution, transition probability, emission probability.
    :param X: Sequence of elements of length T.
    :return: beta. Dict of log-probabilities, where each key (t, z) denotes
        that any particular state z is chosen at the step t.
    """

    pi, tr, em = theta
    # Implementation here
    beta = {}  # (t, z_i) -> log_p
    h_states = pi.keys()
    T = len(X) - 1
    for t in range(len(X) - 1, -1, -1):
        if t == T:
            for hidden_state in h_states:
                beta[(t, hidden_state)] = 1
        else:
            for current_state in pi.keys():
                prob = sum([tr[(current_state, next_state)] * em[(X[t + 1], next_state)] * beta[(t + 1, next_state)] for
                            next_state in pi.keys()])
                beta[(t, current_state)] = prob

    return beta
