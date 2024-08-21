import numpy as np
import matplotlib.pyplot as plt




def TEST_playing_with_tsne_scatter():
    from sklearn.manifold import TSNE

    x = np.load('dataset/extracted-features/images.npy')
    y = np.load('dataset/extracted-features/labels.npy')


    tsne = TSNE(n_components=2, verbose=1, n_jobs=4)
    x_viz = tsne.fit_transform(x)

    print(x_viz.shape)


    plt.scatter(x_viz[y[:, 1] == 0, 0], x_viz[y[:, 1] == 0, 1], c=y[y[:, 1] == 0, 2], cmap='cool')
    plt.scatter(x_viz[y[:, 1] == 1, 0], x_viz[y[:, 1] == 1, 1], c=y[y[:, 1] == 1, 2], cmap='hot')
    plt.colorbar()

    plt.show()

# VIZ: DATASET -> VIZ: KNN (matriz de confusão OU tsne) -> VIZ: K-MEANS (tsne OU? imagens representativas do cluster)
# TALVEZ IGNORAR OUTRO LABEL -- SÓ GATO E CACHORRO


def TEST_KNN_TSNE():
    from sklearn.manifold import TSNE
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    x = np.load('dataset/extracted-features/images.npy')
    y = np.load('dataset/extracted-features/labels.npy')

    print(y)

    # usando só GATO/CACHORRO
    y = y[:, [0, 1]]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=True, random_state=1917)

    knn = KNeighborsClassifier(n_neighbors=37)
    knn.fit(x_train, y_train[:, 1])

    preds = knn.predict(x_test)

    print(classification_report(y_test[:, 1], preds))

    tsne = TSNE(n_components=2, verbose=1, n_jobs=6)
    x_viz = tsne.fit_transform(x_test)    # poderia usar x_train?

    plt.scatter(x_viz[:, 0], x_viz[:, 1], c=y_test[:, 1])
    plt.show()

    plt.scatter(x_viz[:, 0], x_viz[:, 1], c=preds)
    plt.show()

    print(y_test[preds != y_test[:, 1], 0])
    # [2361 4718 1211 525 2579 7027 5963]


def GET_INDEX_WRONGS():
    from data import OxfordPetsDataset
    from torchvision.transforms.functional import to_pil_image

    WRONG_INDEXES = [2361, 4718, 1211, 525, 2579, 7027, 5963]
    ds = OxfordPetsDataset()

    for idx in WRONG_INDEXES:
        to_pil_image(ds[idx][0]).show()




def TEST():
    from sklearn.manifold import TSNE

    x = np.load('dataset/extracted-features/images.npy')
    y = np.load('dataset/extracted-features/labels.npy')

    tsne = TSNE(n_components=2, verbose=1, n_jobs=4)
    x_viz = tsne.fit_transform(x)

    print(x_viz.shape)

    plt.scatter(x_viz[y[:, 1] == 0, 0], x_viz[y[:, 1] == 0, 1], c=y[y[:, 1] == 0, 1], cmap='cool')
    plt.scatter(x_viz[y[:, 1] == 1, 0], x_viz[y[:, 1] == 1, 1], c=y[y[:, 1] == 1, 1], cmap='hot')

    plt.show()



if __name__ == '__main__':
    TEST_KNN_TSNE()    

    # GET_INDEX_WRONGS()