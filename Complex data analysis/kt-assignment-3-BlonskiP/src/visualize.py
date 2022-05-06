import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def plot_series(df,series_name):
    fig = px.line(df, y=series_name, title=f"series: {series_name}")
    fig.show()

def plot_all_series(df):
    fig = px.line(df,title=f"All series")
    fig.show()

def show_hist(df):
    df.hist()
    fig = px.histogram(df)
    fig.show()

def draw_changes(df,series_name,change_ids,means=None):
    fig = go.Figure(data=go.Scatter(x=df.index, y=df[series_name],name=series_name))
    y_max=df[series_name].max()
    y_min=df[series_name].min()
    if means is not None:
        while len(means)>len(df[series_name]):
            means.insert(0, means[0])
        fig.add_trace(go.Scatter(x=df.index, y=means,name='means',line=dict(color="yellow")))

    for point in change_ids:
        # Add shapes
        fig.add_shape(type="line",
                      x0=point, y0=y_min, x1=point, y1=y_max,
                      line=dict(color="red", width=2)
                      )

    fig.show()