from src.hmm import train
from src.hmm.forward_backward import forward, backward

from copy import deepcopy

class BaumWelchAlgorithm(train.TrainingAlgorithm):
    
    def estimate_parameters(self, theta, X):
        """Estimate parameters of the HMM usng Baum-Welch algorithm.

        :param theta: Parameters of the HMM. Tuple of 3 elements: initial
            state distribution, transition probability, emission probability.
        :param X: Sequence of elements of length T.
        :return: Tuple of parameters of the HMM: initial state probability pi,
            transition probability matrix a, and emission probabiliy matrix b.
        """

        gamma = []
        ksi = []
        pi, tr, em = deepcopy(theta)
        
        
        h_states = pi.keys()

        # perform expectation step: compute  𝜉(𝑟)  and  𝛾(𝑟)  for current parameters  𝜃
        for x in X:

            alpha = forward(theta, x)
            beta = backward(theta, x)

            gamma_x = {}
            ksi_x = {}

            for t in range(len(x)-1):

                #gamma
                for hidden_state in h_states:

                    gamma_nominator = alpha[(t, hidden_state)] * beta[(t, hidden_state)]

                    gamma_denominator = 0
                    for other_hidden_state in h_states:
                        gamma_denominator += alpha[(t, other_hidden_state)] * beta[(t, other_hidden_state)]

                    gamma_x[(t,hidden_state)] = gamma_nominator / gamma_denominator

                #ksi
                for hidden_state_i in h_states:
                    for hidden_state_j in h_states:

                        ksi_nominator = alpha[(t, hidden_state_i)]
                        ksi_nominator *= tr[(hidden_state_i, hidden_state_j)]
                        ksi_nominator *= beta[(t+1, hidden_state_j)]
                        ksi_nominator *= em[(x[t+1], hidden_state_j)]

                        ksi_denominator = 0
                        for hidden_state_u in h_states:
                            for hidden_state_v in h_states:
                                w = alpha[(t, hidden_state_u)]
                                w *= tr[(hidden_state_u, hidden_state_v)]
                                w *= beta[(t+1, hidden_state_v)]
                                w *= em[(x[t+1], hidden_state_v)]
                                ksi_denominator += w

                        ksi_x[(t, hidden_state_i, hidden_state_j)] = ksi_nominator / ksi_denominator

            gamma.append(gamma_x)
            ksi.append(ksi_x)

        # perform maximization step: compute new parameters  𝜃  using  𝜉  and  𝛾

        # PI
        for hidden_state, init_prob in pi.items():
            
            nominator = 0
            for gamma_x in gamma:
                nominator += gamma_x[(0, hidden_state)]
            
            pi[hidden_state] = nominator / len(X)

        #TR
        for key, value in tr.items():

            nominator = 0
            for ksi_x in ksi:
                for t in range(len(x)-1):
                    nominator += ksi_x[(t, key[0], key[1])]

            denominator = 0
            for gamma_x in gamma:
                for t in range(len(x)-1):
                    denominator += gamma_x[(t, key[0])]
            
            tr[key] = nominator / denominator

        #EM
        for key, value in em.items():

            nominator = 0
            for index, gamma_x in enumerate(gamma):
                for t in range(len(x)-1):
                    if X[index][t] == key[0]:
                        nominator += gamma_x[(t, key[1])]

            denominator = 0
            for gamma_x in gamma:
                for t in range(len(x)-1):
                    nominator += gamma_x[(t, key[1])]
        #Return model with parameters  𝜃
        result =  (pi, tr, em)
        #print(result)
        return result