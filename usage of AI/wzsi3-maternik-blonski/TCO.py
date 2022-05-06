import random
from Visualization import epoch_result_cols
import numpy as np
import pandas as pd



class TCO:
    #https://ieeexplore.ieee.org/document/5507009
    def __init__(self, objective_function, bounds, agents_size, epochs):
        self.DIM = 2
        self.epochs = epochs
        self.num_termites = agents_size
        self.Best_score = float("inf")
        self.bounds = bounds

        self.lower_bound = []
        self.lower_bound.append(bounds[0][0])
        self.lower_bound.append(bounds[1][0])

        self.upper_bound = []
        self.upper_bound.append(bounds[0][1])
        self.upper_bound.append(bounds[1][1])


        self.history={
            'epochs_resuts': pd.DataFrame(epoch_result_cols),
            'best' : []
        }
        self.obj_function = objective_function
        self.Positions = np.zeros((self.num_termites, self.DIM))

        self.tau = np.zeros(self.num_termites)
        self.rho = 0.8
        self.fitness = np.zeros(self.num_termites)

    def fit(self, positions):
        return -self.obj_function(np.transpose(positions))

    def random_walk(self, tau, pos_curr):
        r1 = random.random() * tau
        r2 = random.random() * tau
        random_w = [r1, r2]
        #print(f"random walk {random_w}")
        return random_w

    def run(self):
        self.Positions = np.zeros((self.num_termites, self.DIM))
        for i in range(self.DIM):
            rand = np.random.uniform(0, 1, self.num_termites)
            bounds = self.upper_bound[i] - self.lower_bound[i]
            add = self.lower_bound[i]
            self.Positions[:, i] = rand * bounds + add
        self.step_max = bounds / 2
        self.step_min = bounds / 8
        # tau initialization
        self.tau = np.zeros(self.num_termites)

        for l in range(0, self.epochs):
            #compute fitness of the termites
            fitness = self.fit(self.Positions)

            #compute tau of the termites
            #self.tau = (1-self.rho) * self.tau + 1/(fitness + 1)
            self.tau = (1-self.rho) * self.tau + fitness

            #compute exploration step size
            self.step_size = ((self.epochs - l) / self.epochs) * (self.step_max - self.step_min) + self.step_min

            self.tau_max = np.max(self.tau)
            self.tau_argmax = np.argmax(self.tau)

            self.pos_best = self.Positions[self.tau_argmax]
            self.best = self.obj_function([[self.pos_best[0]], [self.pos_best[1]]])
            #print(f"{self.tau_argmax}\t{self.best}\t{self.pos_best}\t{self.step_size}")
            self.history['best'].append(self.best)
            #self.history['best'].append(self.tau_max)

            for i in range(0, self.num_termites):
                epoch_result_cols = {'x': [self.Positions[i][0]],
                                     'y': [self.Positions[i][1]],
                                     'fitness': self.obj_function([[self.Positions[i][0]],[self.Positions[i][1]]]),
                                     'epoch': l}
                epoch_res = pd.DataFrame(epoch_result_cols)
                self.history['epochs_resuts'] = self.history['epochs_resuts'].append(epoch_res)

                tau_curr = self.tau[i]
                if tau_curr < self.tau_max:
                    wb = random.random() + 1
                    rb = random.random()
                    pos_curr = self.Positions[i]
                    v = self.pos_best - pos_curr
                    move = wb * rb * v
                    self.Positions[i] += move
                else:
                    pos_curr = self.Positions[i]
                    #self.Positions[i] += self.random_walk(self.step_size, pos_curr)
        self.best = self.obj_function([[self.pos_best[0]], [self.pos_best[1]]])
        return (self.pos_best, self.best)