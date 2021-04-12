from torch import nn
from models.BasicModule import BasicModule

FEATURES_NUM = 70
class DNN(BasicModule):
    #继承基本模型的类
    def __init__(self):
        super(DNN, self).__init__()
        self.dnn = nn.Sequential(
            nn.Linear(FEATURES_NUM, 128),   #第一层输入
            nn.ReLU(inplace=True), #inplace为True，将会改变输入的数据 ，产生新的输出
            nn.Dropout(0.3),
            nn.BatchNorm1d(128),
            nn.Linear(128, 128),  #第2层
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, 128),  #第3层
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, 128),  #第4层
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.BatchNorm1d(128),
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 10),
            nn.ReLU(inplace=True)
        )
        self.fc_it = nn.Sequential(
            nn.Linear(10, 1),
            nn.Sigmoid()
        )

    def forward(self, t_input):
        x = self.dnn(t_input)
        return x