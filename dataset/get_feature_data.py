import os
import torch
from numpy import genfromtxt
from torch.utils import data
import numpy as np
from sklearn.preprocessing import StandardScaler

print(__file__)

# 读入正负样本
def get_data(data_PATH):
    # 两类特征向量输入
    ALL_data = genfromtxt(data_PATH, delimiter=',')
    pos_num = len(ALL_data)
    pos_data = ALL_data[0:0.5*(pos_num),:]
    neg_data = ALL_data[0.5*(pos_num):(pos_num),:]
    return pos_data, neg_data

# 构建数据集
def get_feature_dataset(data_PATH):
    pos_x, neg_x = get_data(data_PATH)
    # 赋予正负样本标签向量列
    pos_y, neg_y = np.zeros((1, len(pos_x))), np.ones((1, len(neg_x)))
    all_y = np.append(pos_y, neg_y)
    all_y = np.transpose([all_y])
    # 合并正负样本矩阵
    all_x = np.concatenate((pos_x, neg_x), axis=0)
    # 标准化
    scaler = StandardScaler().fit(all_x)
    all_x = scaler.transform(all_x)
    # 转为tensor
    dataset = torch.utils.data.dataset.TensorDataset(torch.Tensor(all_x), torch.Tensor(all_y))
    return dataset