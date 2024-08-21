import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from src.plots_knn import plot_TSNE
from src.plots_knn import plot_confusion_matrix
from src.plots_knn import plot_samples
import pandas as pd

#  importando os dados
images = np.load("./dataset/extracted-features/images.npy")
labels = pd.read_csv('dataset/extracted-features/labels.csv')
labels = labels.drop(columns= labels.columns.difference(['species_id', 'breed_id', 'breed_id_on_species']))
labels = labels.values
# print(labels.shape)
# print(np.unique(labels[:,0]))
# print(np.unique(labels[:,1]))
# print(np.unique(labels[:,2]))
# print(np.unique(labels[:,3]))

# dividindo os dados de treino e teste
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.25, shuffle=True, random_state = 42)

# lista contendo os k's que serão usados para o experimento
ks = [2, 37]
# lista das acuracias calculadas para cada k
train_accuracy = []
test_accuracy = []

for k in ks:
    if k == 2:
        y_train_2 = y_train[:, 1]
        y_test_2 = y_test[:,1]
        show = True
    elif k == 37:
        show = False
        y_train_2 = y_train[:, 0]
        y_test_2 = y_test[:,0]

    # inicializando o knn
    knn = KNeighborsClassifier(n_neighbors=k)
    # atualizando o knn para que haja multiplas saidas
    # classifier = MultiOutputClassifier(knn, n_jobs=8)

    # treinando
    knn.fit(x_train, y_train_2)
    # calculando a acuracia para o conjunto de treinamento
    y_pred_train = knn.predict(x_train)
    train_accuracy.append((y_pred_train==y_train_2).mean())

    # calculando as predições e a acuracia para o conjunto de teste
    y_pred_test = knn.predict(x_test)
    print(classification_report(y_test_2, y_pred_test))
    test_accuracy.append((y_pred_test == y_test_2).mean())


    if show:
        plot_TSNE(x_test, y_test_2, y_pred_test)
        plot_confusion_matrix(confusion_matrix(y_test_2, y_pred_test), k)
        wrong_idx = y_test[y_pred_test!=y_test_2, 0]
        plot_samples(wrong_idx)


print(f"Acuracia treino: \nk=2: {train_accuracy[0]}\nk=37: {train_accuracy[1]}")
print(f"\nAcuracia teste:\nk=2: {test_accuracy[0]}\nk=37: {test_accuracy[1]}")