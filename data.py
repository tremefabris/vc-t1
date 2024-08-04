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
        
        self.labels = self.full_labels[['species', 'label']].values - 1     # -1 due to dataset intricacies
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

        ANNOTATION_PATH = os.path.join(self.dataset_root, _annotation_path)

        annotations = (pd.read_csv(ANNOTATION_PATH, sep=' ', header=None, comment='#')
                        .drop([3], axis='columns'))
        annotations.columns = ['name', 'label', 'species']

        annotations['breed'] = (annotations['name']
                                .transform(lambda x: x.strip('_0123456789').lower())  # que coisa feia que eu fiz....
                                .transform(lambda x: ' '.join(x.split('_')) if '_' in x else x))
                                # será que deus ainda me ama depois disso.....

        annotations['breed_index'] = (annotations['name']
                                      .transform(lambda x: int(x.split('_')[-1])))

        annotations['image_path'] = self.dataset_root + _image_folder + annotations['name'] + _image_extension

        return annotations
