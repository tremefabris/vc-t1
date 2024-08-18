import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

#  importando os dados
images = np.load("./dataset/extracted-features/images.npy")
labels = np.load('./dataset/extracted-features/labels.npy')

# dividindo os dados de treino e teste
x_train, x_test, y_train, y_test = train_test_split(images, labels, random_state = 42)

# lista das acuracias calculadas para cada k
train_accuracy = []
test_accuracy = []

# lista de cluster para experimentação
n_clusters = [2, 37]

for cluster in n_clusters:
    kmeans = KMeans(n_clusters=cluster, random_state = 42)
    kmeans.fit(x_train)

    preds = kmeans.predict(x_test)