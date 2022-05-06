#bounds = [(-10,-10), (10,10)]
#boundsPSO = [(-3,3), (-3,3)]
#particle_size = 100
#epoch = 200
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from GWO import GWO
from PSO import PSO
from TCO import TCO
import numpy as np
cols = {
    "algorithm": '',
    'fitness' : []
}
def research(objective_function,pso_bounds=[(-3,3), (-3,3)],gwo_bounds=[(-10,10), (-10,10)],particle_size=100,epoch=100,):
    pso = PSO(objective_function, gwo_bounds, agents_size=particle_size, epochs=epoch)
    gwo = GWO(objective_function, gwo_bounds, agents_size=particle_size, epochs=epoch)
    tco = TCO(objective_function, bounds = gwo_bounds,agents_size=particle_size,epochs=epoch)

    (history_pso, best_pso) = pso.run()
    (history_gwo, best_gwo) = gwo.run()
    (history_tso, best_tso) = tco.run()
    gwo_best_fitness = np.concatenate(gwo.history['best']).ravel().tolist()
    pso_best_fitness = pso.history['best']
    tco_best_fitness =np.concatenate(tco.history['best']).ravel().tolist()
    iter = list(range(epoch))
    #https://plotly.com/python/line-charts/
    fig = go.Figure()
    ## Add trace for gwo
    fig.add_trace(
        go.Scatter(x=iter, y=gwo_best_fitness,
                mode='lines',
                name='GWO'))
    ## Add trace for pso
    fig.add_trace(
        go.Scatter(x=iter, y=pso_best_fitness,
                   mode='lines',
                   name='PSO'))
    ## Add trace for ACO
    #print(tco_best_fitness)
    fig.add_trace(
        go.Scatter(x=iter, y=tco_best_fitness,
                   mode='lines',
                   name='TCO'))
    fig.show()



