import os,torch
from torch.utils import data
import numpy as np
from config import FEATURE_PATH
from config import USE_GPU

print(__file__)
class fusion_feature:
    def __init__(self, model_a, model_b, dataset_a, dataset_b):
        self.modela = model_a
        self.modelb = model_b
        if USE_GPU:
            self.modela.cuda()
            self.modelb.cuda()
        length_a = len(dataset_a)
        length_b = len(dataset_b)
        self.dataa_loader = torch.utils.data.DataLoader(dataset_a, batch_size = length_a, shuffle=False)
        self.datab_loader = torch.utils.data.DataLoader(dataset_b, batch_size = length_b, shuffle=False)

    def get_featurea(self):
        self.modela.eval()
        with torch.no_grad():  # 禁用梯度计算
            for i, (data,label) in enumerate(self.dataa_loader):  # 每个batch
                feature_a = self.modela(data)
        return feature_a

    def get_featureb(self):
        self.modelb.eval()
        with torch.no_grad():  # 禁用梯度计算
            for i, (data,label) in enumerate(self.datab_loader):  # 每个batch
                feature_b = self.modelb(data)
        return feature_b

    def get_feature_csv(self):
        feature_a = np.array(self.get_featurea())
        feature_b = np.array(self.get_featureb())
        fusion_feature = np.hstack((feature_a, feature_b))
        print(fusion_feature.shape)
        np.savetxt(os.path.join(FEATURE_PATH, "fusion_train_feature.csv"), fusion_feature, delimiter=",")