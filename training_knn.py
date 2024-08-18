import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import confusion_matrix

#  importando os dados
images = np.load("./dataset/extracted-features/images.npy")
labels = np.load('./dataset/extracted-features/labels.npy')

# dividindo os dados de treino e teste
x_train, x_test, y_train, y_test = train_test_split(images, labels, random_state = 42)

# lista contendo os k's que serão usados para o experimento
ks = [2, 37]
# lista das acuracias calculadas para cada k
train_accuracy = []
test_accuracy = []

for k in ks:
    # inicializando o knn
    knn = KNeighborsClassifier(n_neighbors=k)
    # atualizando o knn para que haja multiplas saidas
    classifier = MultiOutputClassifier(knn, n_jobs=8)

    # treinando
    classifier.fit(x_train, y_train)
    # calculando a acuracia para o conjunto de treinamento
    train_accuracy.append(classifier.score(x_train, y_train))

    # calculando as predições e a acuracia para o conjunto de teste
    test_accuracy.append(classifier.score(x_test, y_test))

print(train_accuracy)
print(test_accuracy)