import random
from TestFunctions import ackley_function
from Visualization import epoch_result_cols
import pandas as pd

c1 = 1
c2 = 2
w = 0.85
INIT_FITNESS = -float("inf")
mm=-1

class Particle:
    def __init__(self, bounds):

        self.position=[] # x y
        self.velocity=[] # x y
        self.best_position=[]
        init_fitness = INIT_FITNESS
        self.fittnes_best_position = init_fitness
        self.fitness_position = init_fitness
        self.DIM = 2 # x and y
        for i in range(self.DIM):
            self.position.append(random.uniform(bounds[i][0], bounds[i][1]))
            self.velocity.append(random.uniform(-1,1))
            
    def evaluate(self,objective_function):

        self.fitness_position = objective_function([[self.position[0]], [self.position[1]]]) # get fittnes at acualt particle position
        #if mm == -1:
        if self.fitness_position < mm * self.fittnes_best_position:
            self.best_position = self.position #update particle best
            self.fittnes_best_position = self.fitness_position

    def update_velocity(self, global_best_position):
        for i in range(self.DIM):
            r1 = random.random()
            r2 = random.random()
            personal_velocity = PSO.c1 * r1 * (self.best_position[i] - self.position[i])
            swarm_velocity = PSO.c2 * r2 * (global_best_position[i] - self.position[i])
            self.velocity[i] = w * self.velocity[i] + personal_velocity + swarm_velocity
    
    def update_position(self, bounds):
        for i in range(self.DIM):
            self.position[i] = self.position[i]+self.velocity[i]
            #Upperbound check
            if self.position[i]>bounds[i][1]:
                self.position[i]=bounds[i][1]
            if self.position[i]<bounds[i][0]:
                self.position[i]=bounds[i][0]

class PSO():
    def __init__(self, objective_function, bounds, agents_size, epochs):
        self.fitness_global_best = mm * INIT_FITNESS
        self.global_best_position = None
        self.swarm_particles = []
        for i in range(agents_size):
            self.swarm_particles.append(Particle(bounds))
        self.history = {
            'epochs_resuts': pd.DataFrame(epoch_result_cols),
            'best': []
        }
        self.objective_function = objective_function
        self.bounds = bounds
        self.particle_size = agents_size
        self.epochs = epochs
        PSO.c1 = 1
        PSO.c2 = 2
        PSO.w = 0.85


    def run(self):
        for i in range(self.epochs):
            for j in range(self.particle_size):
                self.swarm_particles[j].evaluate(self.objective_function)
                epoch_result_cols = {'x': [self.swarm_particles[j].position[0]],
                                     'y': [self.swarm_particles[j].position[1]],
                                     'fitness': self.swarm_particles[j].fitness_position,
                                     'epoch': i}
                epoch_res = pd.DataFrame(epoch_result_cols)
                self.history['epochs_resuts'] = self.history['epochs_resuts'].append(epoch_res)
                #if mm == -1:
                if self.swarm_particles[j].fitness_position < self.fitness_global_best:
                    self.global_best_position = list(self.swarm_particles[j].position)
                    self.fitness_global_best = float(self.swarm_particles[j].fitness_position)

            for j in range(self.particle_size):
                self.swarm_particles[j].update_velocity(self.global_best_position)
                self.swarm_particles[j].update_position(self.bounds)
            self.history['best'].append(self.fitness_global_best)
            PSO.c1= PSO.c1-PSO.c1/(self.epochs)
            #self.history.append(self.fitness_global_best)
        result = (self.global_best_position, self.fitness_global_best)
        return result