import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy
import plotly.graph_objects as go
import numpy as np

import os
from itertools import combinations

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

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
                            y=scores['silhouette'],
                            mode='lines',
                            name='Silhueta',
                            line=dict(color='darkblue')))

    # Melhor coeficiente do Silhouette
    fig.add_vline(x=best_silhouette, line_width=2, line_dash="dash", line_color="blue")

    fig.update_layout(
        title='Silhueta e Davies Bouldin vs Número de Clusters',
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

    pca = PCA(n_components= 2)

    plotable_data = pd.DataFrame(data= pca.fit_transform(features.values), columns=['x', 'y'], index= range(7349))
    centros = pca.transform(model.cluster_centers_)
    centros = pd.DataFrame({'x': centros[:, 0], 'y': centros[:, 1]})

    plot = group.loc[([0,1,2])][(group.loc[([0,1,2])]['breed'] == 'shiba_inu') | (group.loc[([0,1,2])]['breed'] == 'Sphynx') | (group.loc[([0,1,2])]['breed'] == 'saint_bernard')].reset_index(level= 0, drop= False)
    print(plot)

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
    fig = go.Figure()

    # Todos os dados
    fig.add_trace(go.Scatter(
        x=plotable_data['x'],
        y=plotable_data['y'],
        mode='markers',
        marker=dict(
            color='gray',
            symbol='circle',
            size=3
        ),
        text= None,
        name='Dados'
    ))

    for breed_name in plot['breed'].unique():
        breed_data = plot[plot['breed'] == breed_name]
        fig.add_trace(go.Scatter(
            x=breed_data['x'],
            y=breed_data['y'],
            mode='markers',
            marker=dict(
                color=breed_data['color'].iloc[0],
                symbol=breed_data['symbol'].iloc[0],
                size=7
            ),
            text=plot['name'],
            name=breed_name
        ))

    # centro dos clusters
    fig.add_trace(go.Scatter(
        x=centros['x'],
        y=centros['y'],
        mode='markers',
        marker=dict(color='black', size=10, symbol='x'),
        name='Centros dos Clusters',
        text= centros.index
    ))

    # Retas de fronteira dos clusters

    lines = []
    for i, j in combinations([0,1,2], 2):
        
        m = -((centros.loc[j, 'x'] - centros.loc[i, 'x']) / (centros.loc[j, 'y'] - centros.loc[i, 'y']))
        p = ((centros.loc[j, 'x'] + centros.loc[i, 'x'])/2, (centros.loc[j, 'y'] + centros.loc[i, 'y'])/2)

        b = p[1] - m*p[0]

        lines.append({'m': m, 'b': b, 'p': p})

    p_intersec = _intersec(lines)

    for line in lines:
        m = line["m"]
        b = line["b"]

        # Limita as retas até o ponto de intersecção e os extremos dos dados
        x_min = p_intersec[0] if line['p'][0] > p_intersec[0] else plotable_data['x'].min() - 1
        x_max = p_intersec[0] if line['p'][0] < p_intersec[0] else plotable_data['x'].max() + 1

        y_min = p_intersec[1] if line['p'][1] > p_intersec[1] else plotable_data['y'].min() - 1
        y_max = p_intersec[1] if line['p'][1] < p_intersec[1] else plotable_data['y'].max() + 1

        
        x_values = np.array([x_min, x_max])
        y_values = _calc_y(m, b, x_values)

        # Reajusta os valores para dentro dos limites
        if y_values[0] < y_min or y_values[0] > y_max:
            y_values[0] = np.clip(y_values[0], y_min, y_max)
            x_values[0] = (y_values[0] - b) / m

        if y_values[1] < y_min or y_values[1] > y_max:
            y_values[1] = np.clip(y_values[1], y_min, y_max)
            x_values[1] = (y_values[1] - b) / m

        fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', line=dict(dash='dash', width=3, color='black'), showlegend= False))

    fig.update_layout(
        xaxis_title='X',
        yaxis_title='Y',
        width=1920,
        height=1080
    )

    fig.write_html('./graphs/silhoette.html')
    fig.write_image('./graphs/silhouette.png')
