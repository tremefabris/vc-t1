import os
import pandas as pd
from tqdm import tqdm
from PIL import Image



# TODO: adicionar ArgParse pra decidir resize_size e crop_size em linha de comando



# TODO: usar os.path.join pra todas as strings de arquivos -- tornar cross-platform
# TODO: remover aquela abominação que eu criei ali
def pp_load_data_into_dataframe(*,
                                _root: str = 'dataset/oxford-iiit-pet/',
                                _annotation_path: str = 'annotations/list.txt',
                                _image_folder: str = 'images/',
                                _image_extension: str = '.jpg') -> pd.DataFrame:

    ANNOTATION_PATH = os.path.join(_root, _annotation_path)

    annotations = (pd.read_csv(ANNOTATION_PATH, sep=' ', header=None, comment='#')
                     .drop([3], axis='columns'))
    annotations.columns = ['name', 'label', 'species']

    annotations['breed'] = (annotations['name']
                            .transform(lambda x: x.strip('_0123456789').lower())  # que coisa feia que eu fiz....
                            .transform(lambda x: ' '.join(x.split('_')) if '_' in x else x))
                            # será que deus ainda me ama depois disso.....

    annotations['breed_index'] = (annotations['name']
                                  .transform(lambda x: int(x.split('_')[-1])))

    annotations['image_path'] = _root + _image_folder + annotations['name'] + _image_extension

    return annotations


def _create_file_name(image_name: str,
                      image_species: int,
                      image_label: int) -> str:
    
    image_name = image_name.split('_')
    breed_idx = int(image_name.pop())
    image_name = '-'.join(image_name)

    return f'{image_name}_{breed_idx - 1}_{image_species - 1}_{image_label - 1}.jpg'


# TODO: usar os.path.join pra todas as strings de arquivos -- tornar cross-platform
# TODO: checar se a magia de coordenadas é generalizável
def pp_preprocess_and_save_images(data: pd.DataFrame,
                                resize_size: tuple[int, int] = (256, 256),
                                crop_size: tuple[int, int] = (224, 224),
                                convert_to_rgb: bool = True,
                                *,
                                save_folder: str = 'dataset/preprocessed/'):
    assert crop_size[0] < resize_size[0] or \
           crop_size[1] < resize_size[1], "Invalid sizes between crop and resize operations"

    os.makedirs(save_folder, exist_ok=True)

    data = data[['image_path', 'name', 'species', 'label']].values

    # coordinate magic
    CROP_OFFSETS = tuple((r - c) / 2 for r, c in zip(resize_size, crop_size))
    CROP_COORDS = (CROP_OFFSETS[0],
                   CROP_OFFSETS[1],
                   crop_size[0] + CROP_OFFSETS[0],
                   crop_size[1] + CROP_OFFSETS[1])


    # resizing, cropping & saving loop
    for idx, image_path in enumerate(tqdm(data[:, 0], desc='Resizing & cropping images')):

        image_file_name = _create_file_name(data[idx, 1], data[idx, 2], data[idx, 3])
        path_to_save = os.path.join(save_folder, image_file_name)

        with Image.open(image_path) as pil_img:
            if convert_to_rgb:
                pil_img = pil_img.convert('RGB')
            pil_img = pil_img.resize(resize_size)
            pil_img = pil_img.crop(CROP_COORDS)
            pil_img.save(path_to_save)



if __name__ == '__main__':
    data = pp_load_data_into_dataframe()
    pp_preprocess_and_save_images(data)
