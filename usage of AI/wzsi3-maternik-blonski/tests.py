from PSO import PSO
from TestFunctions import ackley_function
import matplotlib.pyplot as plt
import math
from GWO import GWO


particle_size = 100
epoch = 200

best_pos = []
best = 0.

#objective_function = lambda x: ackley_function(x[0], x[1])

def ackley_function_2(x1,x2):
    #returns the point value of the given coordinate
    part_1 = -0.2*math.sqrt(0.5*(x1*x1 + x2*x2))
    part_2 = 0.5*(math.cos(2*math.pi*x1) + math.cos(2*math.pi*x2))
    value = math.exp(1) + 20 -20*math.exp(part_1) - math.exp(part_2)
    #returning the value
    return value

def objective_function_(x):
    x = [x[0][0], x[1][0]]
    y = 3*(1-x[0])**2*math.exp(-x[0]**2 - (x[1]+1)**2) - 10*(x[0]/5 - x[0]**3 - x[1]**5)*math.exp(-x[0]**2 - x[1]**2) -1/3*math.exp(-(x[0]+1)**2 - x[1]**2);
    return y

#objective_function = objective_function_
objective_function = ackley_function
bounds = [(-10,10), (-10,10)]
boundsPSO = [(-3,3), (-3,3)]
#pso = PSO(objective_function, bounds, agents_size=10, epochs=50)
#(best_pos, best) = pso.run()
gwo = GWO(objective_function, bounds, agents_size=10,epochs=100)
(best_pos,best) = gwo.run()
print(f"{best_pos}\t{best}")
true_val = ackley_function_2(0, 0)
print(f"{true_val}, {true_val == best}, {true_val - best}")

plt.plot(gwo.history['best'])
plt.show()

print(gwo.history['epochs_resuts'])