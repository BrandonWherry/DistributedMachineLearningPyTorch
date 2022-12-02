"""Trainer Module to assist with Distributed Data Parallel training (DDP).

    Built off of official pytorch documentation from ->


"""
import os
import torch
from time import time
from typing import Callable
from torch.utils.data import DataLoader
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP


class Trainer:
    """Trainer Class to assist with DDP training
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        valid_data: DataLoader,
        loss_func: Callable,  # from torch.nn.functional.*
        optimizer: torch.optim.Optimizer,
        max_run_time: float,
        snapshot_name: str,
    ) -> None:
        # Torchrun assigns many environment variables
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.model = model.to(self.local_rank)
        self.train_data = train_data
        self.valid_data = valid_data
        self.loss_func = loss_func
        self.optimizer = optimizer
        # Hours to seconds, training will stop at this time
        self.max_run_time = max_run_time * 60**2
        self.save_path = "training_saves/" + snapshot_name

        self.epochs_run = 0  # current epoch tracker
        self.run_time = 0.0  # current run_time tracker
        self.train_loss_history = list()
        self.valid_loss_history = list()
        self.epoch_times = list()
        self.lowest_loss = np.Inf
        self.train_loss = np.Inf
        self.valid_loss = np.Inf
        if os.path.exists(self.save_path):
            print("Loading snapshot")
            self._load_snapshot(self.save_path)
        # Key DDP Wrapper, this allows Distributed Data Parallel Training on model
        self.model = DDP(self.model, device_ids=[self.local_rank])

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.local_rank}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        self.run_time = snapshot['RUN_TIME']
        self.train_loss_history = snapshot['TRAIN_HISTORY']
        self.valid_loss_history = snapshot['VALID_HISTORY']
        self.epoch_times = snapshot['EPOCH_TIMES']
        self.lowest_loss = snapshot['LOWEST_LOSS']
        print(f"Resuming training from save at Epoch {self.epochs_run}")

    def _calc_validation_loss(self, source, targets) -> float:
        self.model.eval()
        output = self.model(source)
        loss = self.loss_func(output, targets)
        self.model.train()
        return float(loss.item())

    def _run_batch(self, source, targets) -> float:
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = self.loss_func(output, targets)
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def _run_epoch(self):
        b_sz = len(next(iter(self.train_data))[0])
        print(
            f"\n[GPU{self.global_rank}] Epoch: {self.epochs_run} | Batch_SZ: {b_sz} ", end="")
        print(
            f"| Steps: {len(self.train_data)} ", end="")
        print(
            f"| T_loss: {self.train_loss} | V_loss: {self.valid_loss}")
        self.train_data.sampler.set_epoch(self.epochs_run)
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
        self.train_loss, self.valid_loss = self.train_loss_history[-1], self.valid_loss_history[-1]

    def _save_snapshot(self):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": self.epochs_run,
            "RUN_TIME": self.run_time,
            "TRAIN_HISTORY": self.train_loss_history,
            "VALID_HISTORY": self.valid_loss_history,
            "EPOCH_TIMES": self.epoch_times,
            "LOWEST_LOSS": self.lowest_loss
        }
        torch.save(snapshot, self.save_path)
        print(f"Training snapshot saved at {self.save_path}")

    def train(self):
        for _ in range(self.epochs_run, self.epochs_run + 1000):
            start = time()
            self._run_epoch()
            elapsed_time = time() - start
            self.run_time += elapsed_time
            self.epoch_times.append(elapsed_time)
            start = time()
            self.epochs_run += 1
            if self.valid_loss_history[-1] < self.lowest_loss:
                self.lowest_loss = self.valid_loss_history[-1]
                self._save_snapshot()
            elapsed_time = time() - start
            self.run_time += elapsed_time
            self.epoch_times[-1] += elapsed_time
            print(
                f'Train time: {(self.run_time//60**2):.2f} hr {((self.run_time%60.0**2)//60.0):.2f} min')
            if (self.run_time > self.max_run_time):
                print(
                    f"Training completed -> Total train time: {self.run_time:.2f} seconds")
                break

        # Saving import metrics to analyze training on local machine
        if (self.global_rank == 0):
            train_metrics = {
                "EPOCHS_RUN": self.epochs_run,
                "RUN_TIME": self.run_time,
                "TRAIN_HISTORY": self.train_loss_history,
                "VALID_HISTORY": self.valid_loss_history,
                "EPOCH_TIMES": self.epoch_times,
                "LOWEST_LOSS": self.lowest_loss
            }
            torch.save(train_metrics, self.save_path[:-3] + "_metrics.pt")
