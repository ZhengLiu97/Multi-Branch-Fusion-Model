import torch
from torch import optim
import numpy as np
import time
from config import USE_GPU, VAL_RATIO
from utils.persistence import save_py_obj
from utils.visualization import draw_line_graph_by_list

class PreTrainer:
    """预训练器，输入训练集、验证集,测试集（已经划分完善的），选择最优模型结构并保存模型参数"""
    def __init__(self, model, traindataset, testdataset,criterion, eval_func, n_epoch, n_batch=20):
        self.model = model
        if USE_GPU:
            self.model.cuda()
        self.criterion = criterion
        self.eval_func = eval_func
        self.n_batch = n_batch
        self.n_epoch = n_epoch
        self.optimizer = optim.Adam(self.model.parameters())
        val_size = int(VAL_RATIO * len(traindataset)) # 给定一个验证集比例VAL_RATIO
        train_size = len(traindataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(traindataset, [train_size, val_size])
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=n_batch, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=n_batch)
        self.test_loader = torch.utils.data.DataLoader(testdataset)

    def train(self):
        # 定义统计量
        train_losses = []  # 训练时的平均损失值们
        train_acces = []
        val_losses = []
        val_accs = []  # 验证时的准确率们
        # 训练过程
        self.model.train()
        for epoch_idx in range(self.n_epoch): # 每个epoch

            print("===============================  EPOCH {:d}  ===============================".format(epoch_idx))
            train_loss = 0
            time_all = 0
            for batch_idx, (data, label) in enumerate(self.train_loader): # 每个batch

                self.optimizer.zero_grad() # 反向传播梯度设置为0
                t_data = torch.Tensor(data)
                t_label = torch.Tensor(label)
                if USE_GPU:
                    t_data = t_data.cuda()
                    t_label = t_label.cuda()
                since = int(time.time() * 1000)
                # forward + backward + optimize
                predict = self.model(t_data)
                predict = torch.squeeze(predict)
                time_elapsed = int(time.time() * 1000) - since
                loss = self.criterion(predict, t_label)
                loss.backward()
                self.optimizer.step()
                train_loss += float(loss.item())
                time_all += int(time_elapsed)

            print('The train code run {:.0f}ms'.format(time_all / len(self.train_loader) ))
            print('Average Train loss: {:.4f}'.format(train_loss / len(self.train_loader)))
            # 统计量统计
            train_losses.append(train_loss / len(self.train_loader))  # 平均损失
            val_loss,val_acc = self.val()
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            # 保存模型
            print(self.model.save())
        # 画图
        draw_line_graph_by_list(train_losses,val_accs)
        save_py_obj(self.model.model_name + "_train_loss", train_losses)
        save_py_obj(self.model.model_name + "_val_loss", val_losses)
        save_py_obj(self.model.model_name + "_accs", val_accs)

    def val(self):
        self.model.eval()
        val_loss = 0
        time_all = 0
        predict_numpy = np.zeros(0)
        label_numpy = np.zeros(0)
        with torch.no_grad():# 禁用梯度计算
            for batch_idx, (data, label) in enumerate(self.val_loader): # 每个batch
                t_data = torch.Tensor(data)
                t_label = torch.Tensor(label)
                if USE_GPU:
                    t_data = t_data.cuda()
                    t_label = t_label.cuda()
                since = int(time.time() * 1000)
                predict = self.model(t_data)    # forward
                predict = torch.squeeze(predict)
                time_elapsed = int(time.time() * 1000) - since
                predict_numpy = np.append(predict_numpy, predict.cpu().numpy())
                label_numpy = np.append(label_numpy, t_label.cpu().numpy())
                loss = self.criterion(predict, t_label) # forward
                val_loss += loss.item()
                time_all += int(time_elapsed)
            print('The val code run {:.0f}ms'.format(time_all / len(self.val_loader)))
        print("Total VAL Loss: {:.4f}".format(val_loss))
        ACC, TPR, FPR, TNR, PPV, NPV, TN, FP, FN, TP = self.eval_func(label_numpy, predict_numpy)
        print("VAL_ACC: {:.4f}".format(ACC))
        print("VAL_TPR: {:.4f}".format(TPR))
        print("VAL_FPR: {:.4f}".format(FPR))
        print("VAL_TNR: {:.4f}".format(TNR))
        print("VAL_PPV: {:.4f}".format(PPV))
        print("VAL_NPV: {:.4f}".format(NPV))
        print(format(TN))
        print(format(FP))
        print(format(FN))
        print(format(TP))
        return val_loss, ACC

    def test(self):
        self.model.eval()
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
