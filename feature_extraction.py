import torch
import numpy as np

from src.data import OxfordPetsDataset
from src.model import load_resnet18
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", help="Path of dataset",
                    default='data/oxford-iiit-pet/', type=str)
parser.add_argument("--batch_size", help="Amount of batch size",
                    default=64, type=int)
parser.add_argument("--device", help="Device where run program, eg 'cuda' or 'cpu'",
                    default='cuda', type=str)
parser.add_argument("--num_processes", 
                    help="Amount of cores of cpu to run program",
                    default=4, type=int)
args = parser.parse_args()


if __name__ == '__main__':
    path = args.dataset_path
    BATCH_SIZE = args.batch_size
    DEVICE = args.device
    NUM_PROCESSES = args.num_processes


    data_transforms = OxfordPetsDataset.imagenet_transforms()
    dataset = OxfordPetsDataset(dataset_root=path, transform=data_transforms)
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
    

    features = torch.cat(features, dim=0).cpu()
    labels = dataset.full_labels

    np.save('data/extracted-features/images.npy', features)
    labels.to_csv('data/extracted-features/labels.csv')
