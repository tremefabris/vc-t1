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
    if k == 2:
        y_train_2 = y_train[:, 1]
        y_test_2 = y_test[:,1]
    elif k == 37:
        y_train_2 = y_train[:, 2]
        y_test_2 = y_test[:,2]

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
    test_accuracy.append((y_pred_test == y_test_2).mean())


print(f"Acuracia treino: \nk=2: {train_accuracy[0]}\nk=37: {train_accuracy[1]}")
print(f"\nAcuracia teste:\nk=2: {test_accuracy[0]}\nk=37: {test_accuracy[1]}")