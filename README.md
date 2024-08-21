# Visão Computacional - Trabalho 1
Primeiro trabalho da disciplina de Visão Computacional, ministrada pelo Prof. Dr. Cesar H. Comin

Alunos:
- Vitor Lopes Fabris ([tremefabris](https://github.com/tremefabris))
- Jayme Sakae dos Reis Furuyama ([jaymesakae](https://github.com/jaymesakae))
- Vinicius Gonçalves Perillo ([ViniciusPerillo](https://github.com/ViniciusPerillo))

## Informações do Dataset

Fonte: [https://www.robots.ox.ac.uk/~vgg/data/pets/](https://www.robots.ox.ac.uk/~vgg/data/pets/)

![Informações do Oxford-IIIT Pets](markdown/dataset_statistics.jpg)

## Como funciona

Este repositório não contem o dataset, logo, é necessario baixar e passar o caminho o caminho usando a flag assim como no exemplo a seguir

Para extrair as features:

```bash
python3 feature_extration.py --dataset_path path
```

Para executar o treinamento do algoritmo KNN

```bash
python3 training_knn.py
```
