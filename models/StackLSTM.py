from torch import nn
from .BasicModule import BasicModule


NN_INPUT_SIZE = 280
LSTM_HIDDEN_SIZE = 128
LSTM_LEN = 70


class StackLSTM(BasicModule):
    def __init__(self):
        super(StackLSTM, self).__init__()
        # self.embed = nn.Linear(NN_INPUT_SIZE, LSTM_LEN)     # 虽然不知道这有什么用，但还是按需求加上了
        self.lstm = nn.LSTM(
            input_size=LSTM_LEN, #输入特征维数
            hidden_size=LSTM_HIDDEN_SIZE, #隐层状态的维数
            num_layers=3, #lstm层的个数
            bias=True,#隐层状态是否带bias，默认为true
            dropout=0.1,#是否在除最后一个lstm层外的lstm层后面加dropout层
            bidirectional=False, #是否是双向RNN，默认为false
            batch_first=True #是否输入输出的第一维为batchsize
        )
        self.fc = nn.Sequential(
            nn.Linear(LSTM_HIDDEN_SIZE, 1),
            nn.Sigmoid()
        )

    def forward(self, t_input):
        # t_input = self.embed(t_input)
        t_input = t_input.unsqueeze(1)
        t_input, (h_n, c_n) = self.lstm(t_input)
        t_input = self.fc(t_input[:, -1, :])
        return t_input
