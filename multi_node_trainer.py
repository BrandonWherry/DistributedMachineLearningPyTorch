"""DDP Training Script
"""
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


def ddp_setup():
    """Initialzes the backend method for gradient synchronization, and is a key part in
        enabling Distributed Data Parallel training on cuda GPUs. Use 'nccl' for cuda GPU

        process_group creates 1 process per GPU. So a multiGPU system with 4 GPUs will now
        have 4 child processes. Torchrun manages the details of this.
    """
    init_process_group(backend="nccl")


def create_train_objs() -> Tuple[torch.nn.Module, Callable, torch.optim.Optimizer]:
    model = vgg19(weights='IMAGENET1K_V1')
    # Replacing classifier for only 20 outputs, from 1000
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
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    return model, loss_func, optimizer


def create_dataloaders(batch_size: int, data_path: str) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])
    combined_data = ImageFolder(root=data_path, transform=transform)
    train_split = ceil(len(combined_data) * 0.80)
    valid_split = floor(len(combined_data) * 0.20)
    generator = torch.Generator()
    # Each machine must get the same data across all machines
    generator.manual_seed(42)
    train_data, valid_data = random_split(
        combined_data, [train_split, valid_split], generator=generator)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,  # No need for shuffling, as it can mess up the sampler
        # The sampler will make sure that the batches are different across each process in cluster
        sampler=DistributedSampler(train_data)
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=batch_size,
        pin_memory=True,
    )
    return train_loader, valid_loader


def main(max_run_time: float, batch_size: int, snapshot_name: str, data_path='data/train'):
    print('Here1')
    ddp_setup()
    print('Here2')
    train_data, valid_data = create_dataloaders(batch_size, data_path)
    model, loss_func, optimizer = create_train_objs()
    print('Here3')
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
