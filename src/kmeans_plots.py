import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

import os
from itertools import combinations

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

def _intersec(retas: list):
    m1, b1 = retas[0]['m'], retas[0]['b']
    m2, b2 = retas[1]['m'], retas[1]['b']
    m3, b3 = retas[2]['m'], retas[2]['b']
    

    A = np.array([
        [m1, -1],
        [m2, -1],
        [m3, -1]
    ])
    B = np.array([-b1, -b2, -b3])
    
    return np.linalg.lstsq(A, B, rcond=None)[0]

def _calc_y(m, b, x):
        return m * x + b

def best_k_graph(scores:pd.DataFrame, 
                 best_silhouette: int):
    # gráfico
    fig = go.Figure()

    # Silhouette
    fig.add_trace(go.Scatter(x=list(range(2, 38)),
                            y=scores,
                            mode='lines',
                            name='Silhueta',
                            line=dict(color='darkblue')))

    # Melhor coeficiente do Silhouette
    fig.add_vline(x=best_silhouette, line_width=2, line_dash="dash", line_color="blue")

    fig.update_layout(
        #title='Silhueta vs Número de Clusters',
        xaxis=dict(title='Número de Clusters', tickvals=list(range(2, 38, 2))),
        yaxis=dict(title='Silhueta'),
        width=960,
        height=540
    )

    if not os.path.exists('./graphs'):
        os.mkdir('./graphs')

    fig.write_html('./graphs/best_k.html')
    fig.write_image('./graphs/best_k.png')

def silhouette_plot(model: KMeans, 
                    group: pd.DataFrame, 
                    features: pd.DataFrame):

    tsne = TSNE(n_components= 2, n_jobs= 4)
    data = np.vstack((features, model.cluster_centers_))
    features2d = tsne.fit_transform(data)

    plotable_data = pd.DataFrame(data= features2d[:-3], columns=['x', 'y'], index= range(7349))
    centros = pd.DataFrame(data= features2d[-3:], columns=['x', 'y'])

    plot = group.loc[([0,1,2])][(group.loc[([0,1,2])]['breed'] == 'shiba_inu') | (group.loc[([0,1,2])]['breed'] == 'Sphynx') | (group.loc[([0,1,2])]['breed'] == 'saint_bernard')].reset_index(level= 0, drop= True)

    plot = plot.join(plotable_data)

    # Cores e simbolos
    breed_colors = {
        'Sphynx': 'blue',
        'shiba_inu': 'green',
        'saint_bernard': 'red',

    }

    clusters_symbols = {
        0: 'circle',
        1: 'square',
        2: 'diamond',

    }
    plot['color'] = plot['breed'].map(breed_colors)
    plot['symbol'] = plot['silhouette'].map(clusters_symbols)

    # Plot
    fig = px.scatter(group.reset_index(level= 0, drop= True).join(plotable_data), 
                     x='x',
                     y='y',
                     color= 'silhouette')

    fig.update_layout(
        xaxis_title='X',
        yaxis_title='Y',
        width=960,
        height=540
    )

    fig.write_html('./graphs/silhoette.html')
    fig.write_image('./graphs/silhouette.png')

def species_plot(model: KMeans, 
                    group: pd.DataFrame, 
                    features: np.ndarray):

    tsne = TSNE(n_components= 2, n_jobs= 4)
    data = np.vstack((features, model.cluster_centers_))
    features2d = tsne.fit_transform(data)

    plotable_data = pd.DataFrame(data= features2d[:-2], columns=['x', 'y'], index= range(7349))
    centros = pd.DataFrame(data= features2d[-2:], columns=['x', 'y'])

    plot = group.reset_index(level= 0, drop= True)

    plot = plot.join(plotable_data)

    fig = px.scatter(plot, x= 'x', y='y', color='species')

    fig.update_layout(
        xaxis_title='X',
        yaxis_title='Y',
        width=960,
        height=540
    )

    fig.write_html('./graphs/species.html')
    fig.write_image('./graphs/species.png')