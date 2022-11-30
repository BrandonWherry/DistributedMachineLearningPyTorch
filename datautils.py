import torch
from torch.utils.data import Dataset

class MyTrainDataset(Dataset):
    """Random Dataset meant for testing, obtained from
       https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/datautils.py
    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, size):
        self.size = size
        self.data = [(torch.rand(20), torch.rand(1)) for _ in range(size)]

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.data[index]


def imageNetDataLoader():
    pass