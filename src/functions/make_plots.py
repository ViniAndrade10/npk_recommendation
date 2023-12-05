import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.stats import gaussian_kde


def plot_histogram(df:pd.DataFrame, column:str, qty_simulation):
    kde = gaussian_kde(df[column].values)
    x_lin = np.linspace(min(df[column]), max(df[column]), int(qty_simulation))
    y_lin = kde(x_lin)  

    p5 = np.quantile(df[column], 0.05)
    p50 = np.quantile(df[column], 0.5)
    p95 = np.quantile(df[column], 0.95)

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=df[column], 
            nbinsx=int(qty_simulation/10), 
            histnorm='density',
            opacity=0.6,
            marker=dict(color='red', line=dict(color='black', width=0.3))
            )
        )
    fig.add_trace(
        go.Scatter(
        x=x_lin,
        y=y_lin
        )
    )

    fig.add_shape(go.layout.Shape(
        type='line',
        x0=p5,
        x1=p5,
        y0=0.05,
        y1=0.90,
        yref='paper',
        line=dict(color='black', width=2, dash='dash')
    ))

    fig.add_shape(go.layout.Shape(
        type='line',
        x0=p95,
        x1=p95,
        y0=0.05,
        y1=0.95,
        yref='paper',
        line=dict(color='black', width=2, dash='dash')
    ))

    fig.add_annotation(
        x=p5,
        y=-0.04,
        text="5%",
        showarrow=False,
        font=dict(color="black")
    )

    fig.add_annotation(
        x=p95,
        y=-0.04,
        text="5%",
        showarrow=False,
        font=dict(color="black")
    )

    fig.add_annotation(
        x=p50,
        y=-0.04,
        text="90%",
        showarrow=False,
        font=dict(color="black")
    )

    fig.update_layout(title=f"Simulation Values {column}")

    return fig
