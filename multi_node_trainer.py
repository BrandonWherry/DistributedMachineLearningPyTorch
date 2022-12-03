"""DDP Training Script
"""
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from torchvision.models import vgg19
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from math import floor, ceil
from ddp_trainer import Trainer
from typing import Callable, Tuple
from math import sqrt


def ddp_setup():
    """Initialzes the backend method for gradient synchronization. This is a key part in
        enabling Distributed Data Parallel training on cuda GPUs. Use 'nccl' for cuda GPU

        init_process_group() creates 1 process per GPU. Torchrun manages the details of this.
    """
    init_process_group(backend="nccl")


def create_train_objs() -> Tuple[torch.nn.Module, Callable, torch.optim.Optimizer]:
    """Used to instantiate 3 training objects. Model, loss_func, and Optimizer

    Returns:
        Tuple[torch.nn.Module, Callable, torch.optim.Optimizer]: tuple of model, Loss Function, and Optimizer
    """
    model = vgg19(weights='IMAGENET1K_V1')
    # Replacing classifier with only 20 outputs (from 1000)
    # Classifier will train from scratch, while encoder begins with pretrained weights
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(in_features=25088, out_features=4096, bias=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(p=0.7, inplace=False),
        torch.nn.Linear(in_features=4096, out_features=2048, bias=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(p=0.7, inplace=False),
        torch.nn.Linear(in_features=2048, out_features=20, bias=True)
    )
    loss_func = F.cross_entropy
    world_size = float(os.environ["WORLD_SIZE"])
    learning_rate = 0.0001 * sqrt(world_size)
    print(f"Learning Rate = {learning_rate:.8f}")
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    return model, loss_func, optimizer


def create_dataloaders(batch_size: int, data_path: str
    ) -> Tuple[DataLoader, DataLoader]:
    """Used to instantiate 2 Dataloaders for DDP training.

    Args:
        batch_size (int): batch size of each device
        data_path (str): path to dataset

    Returns:
        Tuple[DataLoader, DataLoader]: tuple of training and validation dataloaders
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
    ])

    combined_data = ImageFolder(root=data_path, transform=transform)
    train_split = ceil(len(combined_data) * 0.80)
    valid_split = floor(len(combined_data) * 0.20)
    generator = torch.Generator()
    generator.manual_seed(42) # Ensures that each gpu has the same validation data

    train_data, valid_data = random_split(
        combined_data, [train_split, valid_split], generator=generator)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        pin_memory=True, # Allocates samples into page-locked memory, speeds up data transfer to GPU
        shuffle=False,  # No need for shuffling, as it can mess up the distributed sampler
        sampler=DistributedSampler(train_data)
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=batch_size,
        pin_memory=True,
    )
    return train_loader, valid_loader


def main(max_run_time: float, batch_size: int, snapshot_name: str, data_path='data/train'):
    ddp_setup()
    train_data, valid_data = create_dataloaders(batch_size, data_path)
    model, loss_func, optimizer = create_train_objs()
    trainer = Trainer(model, train_data, valid_data, loss_func,
                      optimizer, max_run_time, snapshot_name)
    trainer.train()
    destroy_process_group() # cleans up after multigpu training


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='distributed training job')
    parser.add_argument('--train_time', default=0.5, type=float,
                        help='How long do you want to train, in hours (default 30 minutes)')
    parser.add_argument('--model_name', default='model_snapshot.pt',
                        help='Input the save name of model (default: model_snapshot.pt)')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Input batch size on each device (default: 64)')
    args = parser.parse_args()

    main(args.train_time, args.batch_size, args.model_name)
