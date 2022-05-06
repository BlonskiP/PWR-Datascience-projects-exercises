import numpy as np
import plotly.express as px
import plotly.graph_objects as go
epoch_result_cols = {'x' : [],
        'y' : [],
        'fitness' : [],
        'epoch':[]}

def plot3d(mesh_arr, z, df):
    x = mesh_arr[0]
    y = mesh_arr[1]
    trace_func = go.Mesh3d(x=x, y=y, z=z, opacity=1, intensity=z, colorscale='Viridis')
    fig = px.scatter_3d(df, x='x', y='y', z='fitness', animation_frame='epoch', range_x=[-10, 10], range_y=[-10, 10],
                        color='fitness')
    fig.add_trace(trace_func)
    fig.show()


def animate_paths(objective_function, optimalizator, epoch=100, agents_size=10, lb=-10, ub=10, function_points=1000):
    bounds = [(lb, ub), (lb, ub)]
    optim = optimalizator(objective_function, bounds, agents_size=agents_size, epochs=epoch)
    optim.run()

    x1_range = [np.random.uniform(lb, ub) for x in range(function_points)]
    x2_range = [np.random.uniform(lb, ub) for x in range(function_points)]
    x_range_array = [x1_range, x2_range]
    z_range = objective_function(x_range_array)
    results = optim.history['epochs_resuts']

    plot3d(x_range_array, z_range, df=results)
