from torch import nn
from torchvision import models


def load_resnet18(top: bool = False,
                  imagenet_weights: bool = True,
                  eval_mode: bool = True) -> models.ResNet:
    
    WEIGHTS = models.ResNet18_Weights.DEFAULT if imagenet_weights else None   
    model = models.resnet18(weights=WEIGHTS)
    if not top:
        model.fc = nn.Identity()
    if eval_mode:
        model.eval()
    
    return model
