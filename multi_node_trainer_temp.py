"""Base Code obtained from:
    https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multinode.py

    This code is nicely modularized. I have altered from the original.

    Changes I made:
        -snapshots are saved based on the best achieved validation accuracy, and not saved at set epoch numbers.
        -modified to calculate validation loss each epoch
        -snapshots now include current validation loss

    Changes made by Brandon W @UTSA
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
from time import time
import numpy as np
from torchvision.models import vgg19


def ddp_setup():
    """Initialzes the backend method for gradient synchronization, and is a key part in
        enabling Distributed Data Parallel training on cuda GPUs.
    """
    init_process_group(backend="nccl")


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        valid_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        max_run_time: float,
        save_path: str,
    ) -> None:
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.model = model.to(self.local_rank)
        self.train_data = train_data
        self.valid_data = valid_data
        self.optimizer = optimizer
        self.epochs_run = 0    #current epoch tracker
        self.save_path = save_path
        self.run_time = 0.0    #current run_time tracker
        self.max_run_time = max_run_time * 60**2 # Converting to hours
        self.train_loss_history = list()
        self.valid_loss_history = list()
        self.lowest_loss = np.Inf
        if os.path.exists(save_path):
            print("Loading snapshot")
            self._load_snapshot(save_path)
        #Key DDP Wrapper, this allows Distributed Data Parallel Training on the model
        self.model = DDP(self.model, device_ids=[self.local_rank])


    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.local_rank}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        self.run_time = snapshot['RUN_TIME']
        self.train_loss_history = snapshot['TRAIN_HISTORY']
        self.valid_loss_history = snapshot['VALID_HISTORY']
        self.lowest_loss = snapshot['LOWEST_LOSS']
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")


    def _calc_validation_loss(self, source, targets) -> float:
        self.model.eval()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        self.model.train()
        return  float(loss.item())


    def _run_batch(self, source, targets) -> float:
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()
        return float(loss.item())


    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        train_loss = 0
        valid_loss = 0

        # Train Loop
        for source, targets in self.train_data:
            source = source.to(self.local_rank)
            targets = targets.to(self.local_rank)
            train_loss += self._run_batch(source, targets)

        # Calculating Validation loss
        for source, targets in self.valid_data:
            source = source.to(self.local_rank)
            targets = targets.to(self.local_rank)
            valid_loss += self._calc_validation_loss(source, targets)

        # Update loss histories
        self.train_loss_history.append(train_loss/len(self.train_data))
        self.valid_loss_history.append(valid_loss/len(self.valid_data))


    def _save_snapshot(self, epoch: int):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
            "RUN_TIME": self.run_time,
            "TRAIN_HISTORY" : self.train_loss_history,
            "VALID_HISTORY" : self.valid_loss_history,
            "LOWEST_LOSS" : self.lowest_loss
        }

        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")


    def train(self):
        exited_epoch_num = self.epochs_run
        for epoch in range(self.epochs_run, 100000):
            elapsed_time = time()
            self._run_epoch(epoch)
            if self.valid_loss_history[-1] < self.lowest_loss:
                self._save_snapshot(epoch)
                self.lowest_loss = self.valid_loss_history[-1]
            elapsed_time = elapsed_time - time()
            self.run_time += elapsed_time
            if (self.run_time > self.max_run_time):
                print(f"Training completed. Total train time: {self.run_time:.2f}")
                break
            exited_epoch_num += 1

        #Saving import metrics to analyze training on local machine
        train_metrics = {
            "EPOCHS_RUN": exited_epoch_num,
            "RUN_TIME": self.run_time,
            "TRAIN_HISTORY" : self.train_loss_history,
            "VALID_HISTORY" : self.valid_loss_history,
            "LOWEST_LOSS" : self.lowest_loss
        }
        torch.save(train_metrics, 'savedmodels/final_training_metrics.pt')

def load_train_objs():
    model = vgg19(weights=None)
    # Replacing lat layer of vgg19 model with one that has 100 outputs, instead of 1000
    # This is because I'm using just a subset of imageNet (1/10th the full dataset)
    model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=100, bias=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return model, optimizer


def prepare_dataloader(batch_size: int):
    train_data = torch.load('dataloaders/train_data.pt')
    valid_data = torch.load('dataloaders/valid_data.pt')
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(train_data)
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(valid_data)
    )
    return train_loader, valid_loader


def main(max_run_time: float, batch_size: int, snapshot_path: str = "savedmodels/snapshot.pt"):
    ddp_setup()
    model, optimizer = load_train_objs()
    train_data, valid_data = prepare_dataloader(batch_size)
    trainer = Trainer(model, train_data, valid_data, optimizer, max_run_time, snapshot_path)
    trainer.train()
    destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='distributed training job')
    parser.add_argument('max_run_time', type=float, help='How long do you want to train, in hours')
    parser.add_argument('--batch_size', default=32, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    
    main(args.max_run_time, args.batch_size)