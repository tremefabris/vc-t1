import os
import torch
import random
import pandas as pd
from torchvision import transforms
from PIL import Image
from glob import glob
from tqdm import tqdm
from itertools import groupby, islice



def _load_image_file_names(root_dir: str = 'dataset/preprocessed/',
                           ordered: bool = True) -> list[str]:
    file_names = glob(root_dir + '*')
    if ordered:
        file_names = sorted(file_names, key=str.lower)
    return file_names

def _extract_labels_from_file_name(file_name: str,
                                   labels: dict,
                                   index: int):

    file_name = os.path.split(file_name)[-1]
    file_name = file_name.split('_')

    labels['index'].append(index)
    labels['breed'].append(file_name[0])
    labels['breed_index'].append(file_name[1])
    labels['species'].append(file_name[2])
    labels['breed_label'].append(file_name[3][:-4])                 # weird indexing to resolve file extension

def _get_transforms() -> transforms.Compose:
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return t

def _split_train_val(image_file_paths: list[str],
                     num_val_elements: int = 50) -> tuple[list[str], list[str]]:

    train_files = []
    val_files = []


    def __get_breed(image_name: str) -> str:
        return image_name.split('_')[0]


    # magia de agrupamento esotérica -- só pra não usar pandas.....
    breed_groups = groupby(image_file_paths, key=__get_breed)

    for _, files_iter in breed_groups:
        breed_val   = islice(files_iter, 0, num_val_elements)      # validation set from the beginning
        breed_train = islice(files_iter, None, None)

        val_files.extend(breed_val)
        train_files.extend(breed_train)

    return train_files, val_files




# TODO: inserir checagem p/ se num_val_per_breed exceder limites válidos
# TODO: existe uma maneira menos repetitiva de fazer esse processamento duplo? (pra train e val)
def load_data(shuffle: bool = True,
              num_val_per_breed: int = 50,
              image_folder: str = 'dataset/preprocessed/',
              ) -> tuple[tuple[torch.Tensor, pd.DataFrame], tuple[torch.Tensor, pd.DataFrame]]:

    image_file_paths = _load_image_file_names(image_folder)
    train_file_paths, val_file_paths = _split_train_val(image_file_paths,
                                                        num_val_per_breed)

    if shuffle:
        random.shuffle(train_file_paths)
        random.shuffle(val_file_paths)


    train_images = torch.zeros(len(train_file_paths), 3, 224, 224, dtype=torch.float32)
    val_images   = torch.zeros(len(val_file_paths),   3, 224, 224, dtype=torch.float32)

    # como fazer esse assignment duplo sem gerar compartilhamento de memória?
    train_labels = {'index': [],                                    # useless?
                    'breed': [],
                    'breed_index': [],
                    'species': [],
                    'breed_label': []}
    val_labels   = {'index': [],                                    # useless?
                    'breed': [],
                    'breed_index': [],
                    'species': [],
                    'breed_label': []}

    t = _get_transforms()


    for idx, train_file_path in enumerate(tqdm(train_file_paths, desc="Preparing training data")):
        _extract_labels_from_file_name(train_file_path, train_labels, idx)

        with Image.open(train_file_path) as img:
            img = t(img)
            train_images[idx] = img
    
    for idx, val_file_path in enumerate(tqdm(val_file_paths, desc="Preparing validation data")):
        _extract_labels_from_file_name(val_file_path, val_labels, idx)

        with Image.open(val_file_path) as img:
            img = t(img)
            val_images[idx] = img
    
    train_labels = pd.DataFrame(train_labels)
    val_labels   = pd.DataFrame(val_labels)

    return (train_images, train_labels), (val_images, val_labels)
