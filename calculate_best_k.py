import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm


def _compute_scores(model: KMeans, features: pd.DataFrame):
    clusters_labels = model.fit_predict(features)
    sc_score = silhouette_score(features, clusters_labels, metric='euclidean')
    return sc_score


def test_k():
    features = np.load("./data/extracted-features/images.npy") 

    silhouette_coefficients = []

    t = 100

    for k in range(2,38):
        sc_mean = 0.0

        model = KMeans(k, init= 'k-means++', n_init= 'auto')

        results = []
        for _ in tqdm(range(t), desc=f"Calculating for K = {k}"):
            results.append(_compute_scores(model, features))

        sc_mean = sum(results) / t
        
        silhouette_coefficients.append(sc_mean/t)

    pd.DataFrame({'silhouette': silhouette_coefficients}).to_csv('./data/extracted-features/clustering_scores.csv', index= False)


if __name__ == '__main__':
    print("This may take a while...")
    test_k()
