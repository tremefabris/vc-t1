import torch
import numpy as np

from data import OxfordPetsDataset
from model import load_resnet18
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", help="Amount of batch size",
                    default=64, type=int)
parser.add_argument("--device", help="Device where run program, eg 'cuda' or 'cpu'",
                    default='cuda', type=str)
parser.add_argument("--num_processes", 
                    help="Amount of cores of cpu to run program",
                    default=4, type=int)
args = parser.parse_args()


# TODO: adicionar ArgParser para definir hyperparams em CLI
if __name__ == '__main__':
    BATCH_SIZE = args.batch_size
    DEVICE = args.device
    NUM_PROCESSES = args.num_processes
    # BATCH_SIZE = 64
    # DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # NUM_PROCESSES = 4


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
    

    features = torch.cat(features, dim=0).cpu()
    labels   = torch.cat(labels, dim=0).cpu()

    print(features.shape)
    print(labels.shape)

    np.save('dataset/extracted-features/images.npy', features)
    np.save('dataset/extracted-features/labels.npy', labels)
