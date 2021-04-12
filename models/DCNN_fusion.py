import torch
from torch import nn
from models.BasicModule import BasicModule
# 定义网络结构
class DCNN(BasicModule):
    def __init__(self):
        super(DCNN,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels =16, kernel_size=3, # 卷积核的大小
                      stride = 1,      # 卷积核移动步长
                      padding = 1),    # 是否对输入张量补0
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(4,stride = 4)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3,  # 卷积核的大小
                      stride=1,  # 卷积核移动步长
                      padding=1),  # 是否对输入张量补0
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(32,32,3,1,1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(32,32,2,1,1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(32*40, 128)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(1)
        # input.shape:(20,1,425)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # feat = x.detach().clone().numpy()
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x