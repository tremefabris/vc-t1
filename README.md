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

Para o melhor fluxo dos treinamentos e validações das metodologias descritas no artigo, é recomendado seguir os seguintes passos para extração de features e fazer os treinamentos. Além disso, este repositório não contém o dataset, logo, é necessario baixar e passar o caminho usando a flag assim como no exemplo a seguir

Para extrair as features:

```bash
python3 feature_extraction.py --dataset_path path
```

Para executar o treinamento do algoritmo KNN

```bash
python3 knn.py
```

Para calcular o melhor valor de clusters para ser executado o K-Means

```bash
python3 calculate_best_k.py
```

Para executar o treinamento do algoritmo K-Means

```bash
python3 kmeans.py
```


