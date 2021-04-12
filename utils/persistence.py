import os
import pickle
import time
from config import CHECKPOINT_PATH


def save_py_obj(name, input_list):
    timestamp = time.strftime("%m_%d_%H_%M_%S")
    filename = name + "_" + timestamp + ".pkl"
    path = os.path.join(CHECKPOINT_PATH, filename)
    with open(path, mode='wb') as f:
        pickle.dump(input_list, f)
        return path


def load_py_obj(filename):
    path = os.path.join(CHECKPOINT_PATH, filename)
    with open(path, mode='rb') as f:
        return pickle.load(f)