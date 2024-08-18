import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
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
    if cluster == 2:
        y_train_2 = y_train[:, 1]
        y_test_2 = y_test[:,1]
    elif cluster == 37:
        y_train_2 = y_train[:, 2]
        y_test_2 = y_test[:,2]

    kmeans = KMeans(n_clusters=cluster, max_iter =  500,random_state = 42)
    kmeans.fit(x_train)

    preds = kmeans.predict(x_train)
    train_accuracy.append((preds == y_train_2).mean())
    
    preds = kmeans.predict(x_test)
    test_accuracy.append((preds == y_test_2).mean())


print(f"Acuracia treino: \nk=2: {train_accuracy[0]}\nk=37: {train_accuracy[1]}")
print(f"\nAcuracia teste:\nk=2: {test_accuracy[0]}\nk=37: {test_accuracy[1]}")