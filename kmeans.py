import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy
import numpy as np

import os
import shutil

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from tqdm import tqdm

from src.kmeans_plots import *

import warnings
warnings.filterwarnings("ignore")


def __copy_image(file: str, cluster: int, metric: str):
    shutil.copy(f'./{file}', f'{file.replace("oxford-iiit-pet/images", f"clusters/{metric}/c{cluster}")}')

def _create_clusters_image_folders(data: pd.DataFrame, metric: str):
    for cluster in tqdm(data.index.levels[0], desc="Creating cluster images folder"):
        group = data.loc[(cluster)]
        if not os.path.exists(f'./data/clusters/{metric}/c{cluster}'):
            os.makedirs(f'./data/clusters/{metric}/c{cluster}')
        else:
            shutil.rmtree(f'./data/clusters/{metric}/c{cluster}')
            os.makedirs(f'./data/clusters/{metric}/c{cluster}')

        group['image_path'].apply(__copy_image, args= [cluster, metric])

def _create_log(data: DataFrameGroupBy, k: int, metric: str):
    total1 = data['breed'].value_counts()
    total2 = data['species_id'].map({1: 'dog', 0: 'cat'}).value_counts()
    
    if not os.path.exists('./log'):
        os.mkdir('./log')

    with open(f'log/{metric}.log', 'w') as sil:
        for i in range(k):
            print_data1 = pd.DataFrame(dict(percentage=((data.loc[(i)]['breed'].value_counts() / total1).dropna()*100),
                                        count= data.loc[(i)]['breed'].value_counts())).sort_values(by='percentage', ascending= False)
            
            print_data2 = pd.DataFrame(dict(percentage=((data.loc[(i)]['species_id'].map({1: 'dog', 0: 'cat'}).value_counts() / total2).dropna()*100),
                                        count= data.loc[(i)]['species_id'].map({1: 'dog', 0: 'cat'}).value_counts())).sort_values(by='percentage', ascending= False)
            
            print(f'Cluster: {i}', file= sil, flush=True)
            print(print_data1, file= sil, flush=True)
            print('', file= sil, flush=True)
            print(print_data2, file= sil, flush=True)
            print('', file= sil, flush=True)
            print('', file= sil, flush=True)


def best_k(scores: pd.Series) -> dict:

    best_silhouette = scores.idxmax() + 2

    best_k_graph(scores, best_silhouette)

    k_param = {
        'species': 2, 
        'silhouette': best_silhouette,
        'breed': 37
    }

    return k_param

def analyze_clusters(scores: pd.Series):
    features = np.load("./data/extracted-features/images.npy")
    labels = pd.read_csv("./data/extracted-features/labels.csv")
    k_param = best_k(scores)

    for metric, k in k_param.items():
        model = KMeans(k, init= 'k-means++', n_init= 'auto', random_state= 257)
        labels[metric] = model.fit_predict(features)
        

        group = labels.groupby(metric, group_keys= True).apply(lambda x: x)
        group['breed'] = group['name'].str.rsplit('_', expand=True, n= 1)[0]

        _create_log(group, k, metric)
        _create_clusters_image_folders(group, metric)

        if metric == 'silhouette':
            silhouette_plot(model, group, features)
        elif metric == 'species':
            species_plot(model, group, features)

        print(f'K-Means com k = {k}')
        print(f'Ajusted Rand Score para label `species`: {adjusted_rand_score(labels["species_id"], labels[metric])}')
        print(f'Ajusted Rand Score para label `breed`: {adjusted_rand_score(labels["breed_id"], labels[metric])}')
        print()


def main():
    scores = pd.read_csv('./data/extracted-features/clustering_scores.csv')['silhouette']
    analyze_clusters(scores)

main()
