import torch
from numpy import genfromtxt
from torch.utils import data
from sklearn.preprocessing import StandardScaler

print(__file__)

# 读入正负样本
def get_data(ALL_data_PATH):
    # 两类特征向量输入
    ALL_data = genfromtxt(ALL_data_PATH, delimiter=',')
    return ALL_data

# 构建数据集
def get_predict_dataset(ALL_data_PATH):
    all_x = get_data(ALL_data_PATH)
    # 标准化
    scaler = StandardScaler().fit(all_x)
    all_x = scaler.transform(all_x)
    # 转为tensor
    dataset = torch.utils.data.dataset.TensorDataset(torch.Tensor(all_x))
    return dataset