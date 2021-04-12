import os
import time
from abc import ABCMeta
import torch
from config import CHECKPOINT_PATH


class BasicModule(torch.nn.Module, metaclass=ABCMeta):
    """封装类，用于提供加载模型及储存模型的方法"""
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = self.__class__.__name__

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self):
        timestamp = time.strftime("%m_%d_%H_%M_%S")
        name = self.model_name + "_" + timestamp + ".pth"
        path = os.path.join(CHECKPOINT_PATH, name)
        torch.save(self.state_dict(), path)
        return name

