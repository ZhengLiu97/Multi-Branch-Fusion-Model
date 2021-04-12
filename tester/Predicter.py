from torch.utils import data
import torch
import time
from config import USE_GPU,threshold

class Predicter:
    """ 预测器，用于有输入，输出最终结果的场景"""
    def __init__(self, model, dataset):
        self.model = model
        if USE_GPU:
            self.model.cuda()
        self.predict_loader = torch.utils.data.DataLoader(dataset)

    def predict(self):
        self.model.load("./checkpoints/DNN_12_10_11_30_23.pth")
        time_all = 0
        with torch.no_grad():  # 禁用梯度计算
            for i, (data) in enumerate(self.predict_loader):  # 每个batch
                t_data = torch.Tensor(data)
                if USE_GPU:
                    t_data = t_data.cuda()
                since = int(time.time() * 1000)
                predict = self.model(t_data)  # forward
                predict_label = (predict >= threshold).astype(int)
                time_elapsed = int(time.time() * 1000) - since
                time_all += int(time_elapsed)
                print("The %d data's label is %d" %(i,predict_label))
            print('The predict code run {:.0f}ms'.format(time_all / len(self.predict_loader)))


