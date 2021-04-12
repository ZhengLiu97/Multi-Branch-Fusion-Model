from torch import nn
from models.BasicModule import BasicModule

class Fusionnet(BasicModule):
    def __init__(self):
        super(Fusionnet, self).__init__()
        self.dnn = nn.Sequential(
            nn.Linear(256, 64),   #第一层输入
            nn.ReLU(inplace=True), #inplace为True，将会改变输入的数据 ，产生新的输出
            nn.Dropout(0.3),
            nn.BatchNorm1d(64),
        )
        self.fc = nn.Sequential(
                nn.Linear(64, 1),
                nn.Sigmoid()
        )

    def forward(self, t_input):
        x= self.dnn(t_input)
        x= self.fc(x)
        return x