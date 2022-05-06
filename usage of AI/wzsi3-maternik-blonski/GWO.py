import random
from Visualization import epoch_result_cols
import numpy as np
import pandas as pd

class GWO:
    def __init__(self, objective_function, bounds, agents_size, epochs):
        # We consider only 2D
        self.DIM = 2
        self.epochs = epochs
        self.num_wolf_pack = agents_size
        # Wolf pack starts at 0.0 (np zeros is list of zeros)
        self.Alpha_Positions = np.zeros(self.DIM)
        self.Beta_Positions = np.zeros(self.DIM)
        self.Delta_Positions = np.zeros(self.DIM)
        # And they have really high score they want to minimalizer
        self.Alpha_score = float("inf")
        self.Beta_score = float("inf")
        self.Delta_score = float("inf")
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
        self.Positions = np.zeros((self.num_wolf_pack, self.DIM))
    def update_wolfs(self, fitness, position):
        if fitness < self.Alpha_score:
            #print('Alpha update')
            self.Delta_score = self.Beta_score  # Update delte
            self.Delta_Positions = self.Beta_Positions.copy()
            self.Beta_score = self.Alpha_score  # Update beta
            self.Beta_Positions = self.Alpha_Positions.copy()
            self.Alpha_score = fitness;  # Update alpha
            self.Alpha_Positions = self.Positions[position, :].copy()

        if (fitness > self.Alpha_score and fitness < self.Beta_score):
           # print('Beta update')
            self.Delta_score = self.Beta_score  # Update delte
            self.Delta_Positions = self.Beta_Positions.copy()
            self.Beta_score = fitness  # Update beta
            self.Beta_Positions = self.Positions[position, :].copy()

        if (fitness > self.Alpha_score and fitness > self.Beta_score and fitness < self.Delta_score):
           ## print('Delta update')
            self.Delta_score = fitness  # Update delta
            self.Delta_Positions = self.Positions[position, :].copy()

    def reset_bounds_position(self, wolf):
        for j in range(self.DIM):
           # print(self.Positions[wolf,j] , self.lower_bound[j] , self.upper_bound[j])
            self.Positions[wolf, j] = np.clip(self.Positions[wolf, j], self.lower_bound[j], self.upper_bound[j])
           # print(self.Positions[wolf, j])
    def run(self):

        self.Positions = np.zeros((self.num_wolf_pack, self.DIM))
        #Get random positions for wolfs
        for i in range(self.DIM):
           # rand = np.random.uniform(0, 1, self.num_wolf_pack)
           # bounds = self.upper_bound[i] - self.lower_bound[i]
           # add = self.lower_bound[i]
           # print(rand, bounds,add)
           # self.Positions[:, i] = rand * bounds + add
           for w in range(0, self.num_wolf_pack):
            pos = random.uniform(self.lower_bound[i], self.upper_bound[i])
            self.Positions[w, i] = pos

            #print(self.Positions[:,i])
        #optim loop
        for l in range(0, self.epochs):

            # update loop
            #print('epoch:',l)
            for i in range(0, self.num_wolf_pack):
                self.reset_bounds_position(i)
                actual_wolf_pos = self.Positions[i, :].tolist()
                #print(actual_wolf_pos)
                fitness = self.obj_function([[actual_wolf_pos[0]], [actual_wolf_pos[1]]])
                epoch_result_cols = {'x': [actual_wolf_pos[0]],
                                     'y': [actual_wolf_pos[1]],
                                     'fitness': fitness,
                                     'epoch': l}
                epoch_res = pd.DataFrame(epoch_result_cols)
                self.history['epochs_resuts'] = self.history['epochs_resuts'].append(epoch_res)
                self.update_wolfs(fitness, i)

            # exploration
            a = 2 - l * (2 / self.epochs)
            self.history['best'].append(self.Alpha_score)
            for i in range(0, self.num_wolf_pack):
                old_pos = self.Positions.copy()
                for j in range(0, self.DIM):
                    r1 = random.random()
                    r2 = random.random()

                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2

                    D_alpha = abs(C1 * self.Alpha_Positions[j] - self.Positions[i, j])
                    X1 = self.Alpha_Positions[j] - A1 * D_alpha

                    r1 = random.random()
                    r2 = random.random()

                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2

                    D_beta = abs(C2 * self.Beta_Positions[j] - self.Positions[i, j])
                    X2 = self.Beta_Positions[j] - A2 * D_beta

                    r1 = random.random()
                    r2 = random.random()

                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2

                    D_delta = abs(C3 * self.Delta_Positions[j] - self.Positions[i, j])
                    X3 = self.Delta_Positions[j] - A3 * D_delta

                    self.Positions[i, j] = (X1 + X2 + X3) / 3
                fit = np.zeros(self.num_wolf_pack)
                for wolf in range(0, self.num_wolf_pack):
                    actual_wolf_pos = self.Positions[i, :].tolist()
                    fit[wolf] = self.obj_function([[actual_wolf_pos[0]], [actual_wolf_pos[1]]])




        return (self.Alpha_Positions, self.Alpha_score)
