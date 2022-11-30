import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from math import floor, ceil

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

combined_data = ImageFolder(root='data/train',
        transform=transform)

test_data = ImageFolder(root='data/validation',
        transform=transform)

traing_split = ceil(len(combined_data) * 0.80)
val_split = floor(len(combined_data) * 0.20)

training_data, validation_data = random_split(combined_data, [traing_split, val_split])

train_loader = DataLoader(combined_data, batch_size=32, shuffle=True)



torch.save(train_loader, 'test.pt')