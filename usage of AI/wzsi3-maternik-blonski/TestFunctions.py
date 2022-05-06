import numpy as np
import math
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px


# returns an array of values for the given x range of values
def ackley_function(x_range_array):
    value = np.empty([len(x_range_array[0])])
    for i in range(len(x_range_array[0])):
        # returns the point value of the given coordinate
        part_1 = -0.2 * math.sqrt(
            0.5 * (x_range_array[0][i] * x_range_array[0][i] + x_range_array[1][i] * x_range_array[1][i]))
        part_2 = 0.5 * (math.cos(2 * math.pi * x_range_array[0][i]) + math.cos(2 * math.pi * x_range_array[1][i]))
        value_point = math.exp(1) + 20 - 20 * math.exp(part_1) - math.exp(part_2)
        value[i] = value_point
    # returning the value array
    return value


def Rastrigin_2D(x_range_array):  # 10*n + sum (x^2_i - 10cos(2pixi)
    #print(x_range_array)
    value = np.empty([len(x_range_array[0])])
    A = 10
    n = 2
    for i in range(len(x_range_array[0])):
        x = (x_range_array[0][i] ** 2) - (A * np.cos(x_range_array[0][i] * 2 * np.pi))
        y = (x_range_array[1][i] ** 2) - (A * np.cos(x_range_array[1][i] * 2 * np.pi))
        value[i] = (A * n) + x + y
    return value


def Visualize_Test_function(obj_fucntion=ackley_function,lb=-10, ub=10,
                            function_points=1000):  # range_x\y shoud be lsit like [-10,10]
    x1_range = [np.random.uniform(lb, ub) for x in range(function_points)]
    x2_range = [np.random.uniform(lb, ub) for x in range(function_points)]
    x_range_array = [x1_range, x2_range]
    z_range = obj_fucntion(x_range_array)
    x = x_range_array[0]
    y = x_range_array[1]
    z = z_range
    fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z, opacity=1, intensity=z, colorscale='Viridis')])
    fig.show()
