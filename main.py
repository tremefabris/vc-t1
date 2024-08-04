import torch
from data import load_data
from model import load_resnet18
from tqdm import tqdm

from time import sleep                                              # for memory debugging


if __name__ == '__main__':

    (train_x, train_y), (val_x, val_y) = load_data(shuffle=True, num_val_per_breed=50)


    train_features = torch.zeros((train_x.shape[0], 512))
    val_features   = torch.zeros((val_x.shape[0], 512))


    model = load_resnet18()


    for idx, train_sample in enumerate(tqdm(train_x, desc="Extracting training features")):
        with torch.no_grad():
            train_features[idx] = model(train_sample.unsqueeze(0))
    
    for idx, val_sample in enumerate(tqdm(val_x, desc="Extracting validation features")):
        with torch.no_grad():
            val_features[idx] = model(val_sample.unsqueeze(0))
