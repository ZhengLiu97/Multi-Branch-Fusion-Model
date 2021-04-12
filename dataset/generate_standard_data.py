import os
from config import DATA_ROOT_PATH
from utils.pretreatment import standard_numpy_array
import numpy as np

BARCELONA_DATA_PATH = os.path.abspath(os.path.join(DATA_ROOT_PATH, "./Bern Barcelona/"))
STANDARD_BARCELONA_DATA_PATH = os.path.abspath(os.path.join(DATA_ROOT_PATH, "./Standard Bern Barcelona"))
# Bonn_DATA_PATH = os.path.abspath(os.path.join(DATA_ROOT_PATH, "./Bonn University/"))
# STANDARD_Bonn_DATA_PATH = os.path.abspath(os.path.join(DATA_ROOT_PATH, "./Standard Bonn University"))
# LINCHUANG_DATA_PATH = os.path.abspath(os.path.join(DATA_ROOT_PATH, "./pt1/"))
# STANDARD_LINCHUANG_DATA_PATH = os.path.abspath(os.path.join(DATA_ROOT_PATH, "./Standard pt1"))

def generate_standard_data():
    n = 0
    sum = 0.0
    for root, _, files in os.walk(BARCELONA_DATA_PATH):
        for name in files:
            path = os.path.join(root, name)
            raw = np.loadtxt(path, delimiter=',', dtype=np.float32)
            sum += (raw[:, 0]).sum()
            # sum += (raw[:]).sum()
            n += raw.shape[0]
    mean = sum / n
    MAGIC_NUMBER = 4396  # 防止浮点数溢出，都减去一个数
    std_sum = 0.0
    for root, _, files in os.walk(BARCELONA_DATA_PATH):
        for name in files:
            path = os.path.join(root, name)
            raw = np.loadtxt(path, delimiter=',', dtype=np.float32)
            data = raw[:, 0]
            # data = raw[:]
            ans = (np.square(data - mean) - MAGIC_NUMBER).sum()
            std_sum += ans
    std = np.sqrt(std_sum / n + MAGIC_NUMBER)
    for root, _, files in os.walk(BARCELONA_DATA_PATH):
        for name in files:
            path = os.path.join(root, name)
            raw = np.loadtxt(path, delimiter=',', dtype=np.float32)
            # 两列取第一列
            data = raw[:, 0]
            # data = raw[:]
            # 读入并且标准化
            std_data = standard_numpy_array(data, mean, std)
            new_dir = os.path.join(STANDARD_BARCELONA_DATA_PATH, os.path.relpath(root,BARCELONA_DATA_PATH))
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
            new_path = os.path.join(new_dir, os.path.splitext(name)[0] + ".npy")
            np.save(new_path, std_data)
generate_standard_data()