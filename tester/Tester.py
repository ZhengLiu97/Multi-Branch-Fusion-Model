from torch.utils import data
import torch
import numpy as np
import time
from config import USE_GPU

class Tester:
    """ 测试器，用于有输入，输出测试性能的场景"""
    def __init__(self, model, dataset,eval_func):
        self.model = model
        if USE_GPU:
            self.model.cuda()
        self.eval_func = eval_func
        self.test_loader = torch.utils.data.DataLoader(dataset)

    def test(self):
        self.model.load("./checkpoints/DNN_12_10_11_30_23.pth")
        time_all = 0
        predict_numpy = np.zeros(0)
        label_numpy = np.zeros(0)
        with torch.no_grad():  # 禁用梯度计算
            for i, (data, label) in enumerate(self.test_loader):  # 每个batch
                t_data = torch.Tensor(data)
                t_label = torch.Tensor(label)
                if USE_GPU:
                    t_data = t_data.cuda()
                    t_label = t_label.cuda()
                since = int(time.time() * 1000)
                predict = self.model(t_data)  # forward
                predict = torch.squeeze(predict)
                time_elapsed = int(time.time() * 1000) - since
                predict_numpy = np.append(predict_numpy, predict.cpu().numpy())
                label_numpy = np.append(label_numpy, t_label.cpu().numpy())
                time_all += int(time_elapsed)
            print('The test code run {:.0f}ms'.format(time_all / len(self.test_loader)))
        ACC, TPR, FPR, TNR, PPV, NPV, TN, FP, FN, TP = self.eval_func(label_numpy, predict_numpy)
        print("TEST_ACC: {:.4f}".format(ACC))
        print("TEST_TPR: {:.4f}".format(TPR))
        print("TEST_FPR: {:.4f}".format(FPR))
        print("TEST_TNR: {:.4f}".format(TNR))
        print("TEST_PPV: {:.4f}".format(PPV))
        print("TEST_NPV: {:.4f}".format(NPV))
        print(format(TN))
        print(format(FP))
        print(format(FN))
        print(format(TP))