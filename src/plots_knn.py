import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import seaborn as sns
from data import OxfordPetsDataset
from torchvision.transforms.functional import to_pil_image


def plot_TSNE(x, y, labels):
    """Função para criar um plot das distribuição após a inferencia

    Parameters
    ----------
    x : Pytorch Tensor
        Tensor contendo as representações dos dados
    y : Pytorch Tensor
        Tensor contendo os rotulos dos dados
    labels : list
        Lista contendo as predições feitas pelo algoritmo
    """
    # Inicializando o TSNE
    classes = len(np.unique(y))
    tsne = TSNE(n_components = classes)
    x_viz = tsne.fit_transform(x)

    # Criando figura da distribuição original dos dados
    fig, ax = plt.subplots()
    scatter = ax.scatter(x_viz[:, 0], x_viz[:,1], c = y)
    legend = ax.legend(*scatter.legend_elements())
    plt.title("Test Distribution")
    plt.show()

    # Criando figura da distribuição dos dados após o treinamento
    fig, ax = plt.subplots()
    scatter = ax.scatter(x_viz[:, 0], x_viz[:,1], c = labels)
    legend = ax.legend(*scatter.legend_elements())
    plt.title(f"K = {classes} distribution")
    plt.show()

    return

def plot_confusion_matrix(data, k):
    """Funcao para demonstrar a matriz de confusao em um mapa de calor

    Parameters
    ----------
    data : list
        Lista contendo a saida do relatorio de classificação do sklearn
    """
    sns.heatmap(data, annot=True, fmt='.0f' ,cmap='PuRd')
    plt.title(f"K = {k} Confusion Matrix")
    plt.show()
    
def plot_samples(ids):
    """Função para mostrar as imagens que não foram classificadas corretamente

    Parameters
    ----------
    ids : List
        Lista contendo os IDS das imagens que foram classificadas incorretamente
    """
    dataset = OxfordPetsDataset()
    plt.figure()
    idx = 1
    for id in ids:
        img, _ = dataset[id]
        img = to_pil_image(img)
        img.show()
    