import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

images = np.load("./dataset/extracted-features/images.npy")
labels = np.load('./dataset/extracted-features/labels.npy')
print(f"len images: {len(images)}")
print(f'shape images: {images.shape}\n')
print(f'len labels: {len(labels)}')
print(f'shape labels: {labels.shape}')

x_train, x_test, y_train, y_test = train_test_split(images, labels, random_state = 42)

ks = [2, 37]
train_accuracy = []
test_accuracy = []
print(f'dataset train: {len(x_train)}')
print(f'dataset test: {len(x_test)}')
for k in ks:
    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(x_train, y_train)

    train_accuracy.append(knn.score(x_train, y_train))

    test_accuracy.append(knn.score(x_test, y_test))

print(train_accuracy)
print(test_accuracy)