import plotly.plotly as py
import plotly.figure_factory as ff
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools

import numpy as np


def perturb(grid=0.1):
    return np.random.uniform(low=-grid, high=grid)/2
    
def scatter_3d(df, x = 0, y = 1, z = 2):
    colors = df.color

    x,y,z = df[x], df[y], df[z]
    trace1 = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        #text=whole_df['artist_name'],
        #name=whole_df['artist_name'],
        showlegend=True,
        marker=dict(
            size=10,
            color=colors,
            colorscale='Jet',
            showscale=True,
            line=dict(
                color=colors,
                width=0.5,
                colorscale='Jet',
            ),

            opacity=1.0
        )
    )

    data = [trace1]
    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )
    fig = go.Figure(data=data, layout=layout)
    return py.iplot(fig, filename='simple-3d-scatter')

def scatter_2d(df, x = 0, y = 1):
    colors = df.color

    x,y = df[x], df[y]
    trace1 = go.Scatter(
        x=x,
        y=y,
        mode='markers',
        #text=whole_df['artist_name'],
        #name=whole_df['artist_name'],
        showlegend=True,
        marker=dict(
            size=10,
            color=colors,
            colorscale='Jet',
            showscale=True,
            line=dict(
                color=colors,
                width=0.5,
                colorscale='Jet',
            ),

            opacity=1.0
        )
    )

    data = [trace1]
    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )
    fig = go.Figure(data=data, layout=layout)
    return py.iplot(fig, filename='simple-3d-scatter')