import os
import torch
import pandas as pd

from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode


class OxfordPetsDataset(Dataset):
    
    def __init__(self,
                 dataset_root = 'dataset/oxford-iiit-pet/',
                 transform = None,
                 target_transform = None):
        self.dataset_root = dataset_root
        self.transform = transform
        self.target_transform = target_transform

        self.full_labels = self.__load_labels()
        
        self.labels = self.full_labels[['species_id', 'breed_id', 'breed_id_on_species']].values
        self.image_files = self.full_labels['image_path'].values


    def __len__(self) -> int:
        return len(self.image_files)


    def __getitem__(self, idx: int):
        image = self.image_files[idx]
        image = read_image(image, mode=ImageReadMode.RGB)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label


    @staticmethod
    def imagenet_transforms():
        from torchvision import transforms

        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD  = [0.229, 0.224, 0.225]

        t = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

        return t


    def __load_labels(self,
                      _annotation_path: str = 'annotations/list.txt',
                      _image_folder: str = 'images/',
                      _image_extension: str = '.jpg') -> pd.DataFrame:
        '''
            Gera o arquivo annotations com informações de labels e o caminho da imagem a partir do arquivo de anotações da base
            Entrada: 
                _annotation_path: caminho do arquivo de anotações da base em string
                _image_folder: caminho para a pasta das imagens em string
                _image_extension: extensão dos aqruvos de imagens em string
            Saída: DataFrame pandas com as colunas: name, breed_id, specie_id, breed_id_on_species, breed, breed_index, image_path
        '''

        ANNOTATION_PATH = os.path.join(self.dataset_root, _annotation_path)
        
        annotations = pd.read_csv(ANNOTATION_PATH, sep=' ', header=None, comment='#', 
                                  names= ['name', 'breed_id', 'species_id', 'breed_id_on_species'])

        #corrige os id de 1:x para 0:x-1
        annotations[['breed_id', 'species_id', 'breed_id_on_species']] -= 1

        breed_and_index = annotations['name'].str.lower().str.rsplit(pat= '_', n=1, expand= True)
        breed_and_index.columns = ['breed', 'breed_index']

        annotations = annotations.join(breed_and_index)

        annotations['image_path'] = self.dataset_root + _image_folder + annotations['name'] + _image_extension

        return annotations