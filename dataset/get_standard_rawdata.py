from .folder import FolderDataset
import numpy as np
import torch

def loader(path):
    return torch.from_numpy(np.load(path))

def target_transform(target):
    return np.float32(target)

def get_standard_rawdataset(Standard_deep_PATH,in_mem=True):
    return FolderDataset(Standard_deep_PATH, loader, in_mem, target_transform=target_transform)