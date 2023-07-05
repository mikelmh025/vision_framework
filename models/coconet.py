# some sort of randomly defined model to demo custom model

import torch
import torch.nn as nn
from torchvision.models import resnet18

class Coconet(nn.Module):
    def __init__(self, num_classes):
        super(Coconet, self).__init__()
        # Define your model architecture here
        
        # Example architecture:
        self.features = resnet18(weights='DEFAULT')
        self.features.fc = nn.Linear(self.features.fc.in_features, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        return x
