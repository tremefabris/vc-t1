import torch
import pandas as pd

from data import OxfordPetsDataset
from model import load_resnet18
from torch.utils.data import DataLoader
from tqdm import tqdm


# TODO: adicionar ArgParser para definir hyperparams em CLI
if __name__ == '__main__':
    
    BATCH_SIZE = 64
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_PROCESSES = 4


    data_transforms = OxfordPetsDataset.imagenet_transforms()
    dataset = OxfordPetsDataset(transform=data_transforms)
    dataloader = DataLoader(dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            num_workers=NUM_PROCESSES)
    
    model = load_resnet18()
    model.to(DEVICE)


    features = []
    labels   = []
    for x, y in tqdm(dataloader, desc="Extracting features"):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        with torch.no_grad():
            features.append(model(x))
        labels.append(y)
    

    features = torch.cat(features, dim=0)
    labels   = torch.cat(labels, dim=0)

    features_df = pd.DataFrame(features.numpy(), columns= range(512)).join(pd.DataFrame(labels.numpy(), columns=['species_id', 'breed_id', 'breed_id_on_species']))
    features_df.to_csv('features.csv', index= False)

    print(features.shape)
    print(labels.shape)
