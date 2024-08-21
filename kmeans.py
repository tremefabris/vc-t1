import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy

import os
import shutil

from joblib import Parallel, delayed

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

from kmeans_plots import *
  

def _compute_scores(model: KMeans, features: pd.DataFrame):
    clusters_labels = model.fit_predict(features)
    sc_score = silhouette_score(features, clusters_labels, metric='euclidean')
    return sc_score


def test_k():
    data = pd.read_csv('features.csv')
    
    features = data.iloc[:,:512]

    silhouette_coefficients = []

    t = 100

    for k in range(2,38):
        sc_mean = 0.0
        db_mean = 0.0

        model = KMeans(k, init= 'k-means++', n_init= 'auto')

        results = Parallel(n_jobs=-1)(delayed(_compute_scores)(model, features) for _ in range(t))

        sc_mean = sum(results) / t
        
        silhouette_coefficients.append(sc_mean/t)

    pd.DataFrame({'silhouette': silhouette_coefficients}).to_csv('clustering_scores.csv', index= False)
    return pd.Series(silhouette_coefficients)

def __copy_image(file: str, cluster: int, metric: str):
    shutil.copy(f'./{file}', f'{file.replace("oxford-iiit-pet/images", f"clusters/{metric}/c{cluster}")}')

def _create_clusters_image_folders(data: pd.DataFrame, metric: str):
    for cluster in data.index.levels[0]:
        group = data.loc[(cluster)]
        if not os.path.exists(f'./dataset/clusters/{metric}/c{cluster}'):
            os.makedirs(f'./dataset/clusters/{metric}/c{cluster}')
        else:
            shutil.rmtree(f'./dataset/clusters/{metric}/c{cluster}')
            os.makedirs(f'./dataset/clusters/{metric}/c{cluster}')

        group['image_path'].apply(__copy_image, args= [cluster, metric])

def _create_log(data: DataFrameGroupBy, k: int):
    total = data['breed'].value_counts()
    
    if not os.path.exists('./log'):
        os.mkdir('./log')

    with open('logs/silhouete.log', 'w') as sil:
        for i in range(k):
            print_data = pd.DataFrame(dict(percentage=((data.loc[(i)]['breed'].value_counts() / total).dropna()*100),
                                        count= data.loc[(i)]['breed'].value_counts())).sort_values(by='percentage', ascending= False)
            
            print(f'Cluster: {i}', file= sil, flush=True)
            print(print_data, file= sil, flush=True)
            print('', file= sil, flush=True)
            print('', file= sil, flush=True)


def best_k(scores: pd.Series) -> dict:
    #scores = pd.read_csv('clustering_scores.csv')

    best_silhouette = scores.idxmax() + 2

    best_k_graph(scores, best_silhouette)

    k_param = {
        'species': 2, 
        'silhouette': best_silhouette,
        'breed': 37
    }

    return k_param

def analise_clusters(scores: pd.Series):
    data = pd.read_csv('features.csv')
    features = data.iloc[:,:512]
    labels = data.iloc[:,512:] 
    k_param = best_k(scores)

    for metric, k in k_param.items():
        model = KMeans(k, init= 'k-means++', n_init= 'auto', random_state= 257)
        labels[metric] = model.fit_predict(features)
        

        group = labels.groupby(metric).apply(lambda x: x, include_groups=False)
        group['breed'] = group['name'].str.rsplit('_', expand=True, n= 1)[0]

        _create_log(group, k)
        # _create_clusters_image_folders(group, metric)

        if metric == 'silhouette':
            silhouette_plot(model, group, features)


def main():
    scores = test_k()
    analise_clusters(scores)

main()

