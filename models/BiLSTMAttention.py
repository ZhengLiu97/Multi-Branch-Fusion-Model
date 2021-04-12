import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

from models.BasicModule import BasicModule
INPUT_LENGTH = 280
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.1
HIDDEN_LINEAR_SIZE = 256
LINEAR_DROPOUT = 0.3
EMBED_DROPOUT = 0.1

class BiLSTMAttention(BasicModule):
    def __init__(self):
        super(BiLSTMAttention, self).__init__()
        self.output_size = 1
        self.hidden_size = LSTM_HIDDEN_SIZE #隐藏层维数
        self.input_length = INPUT_LENGTH #输入特征维数
        # self.embedding = nn.Embedding.from_pretrained(pretrained_weight, freeze=False)# embedding层
        self.embed_dropout = nn.Dropout2d(EMBED_DROPOUT)
        self.lstm = nn.LSTM(input_size=INPUT_LENGTH,
                            hidden_size=LSTM_HIDDEN_SIZE,
                            batch_first = True,
                            num_layers=LSTM_NUM_LAYERS,
                            dropout=LSTM_DROPOUT,
                            bidirectional=True)
        self.linear1 = nn.Linear(self.hidden_size*12, HIDDEN_LINEAR_SIZE)
        self.linear2 = nn.Linear(HIDDEN_LINEAR_SIZE, 1)
        self.attn_fc_layer = nn.Linear(self.hidden_size*2*2*2,HIDDEN_LINEAR_SIZE)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(LINEAR_DROPOUT)

    def attention_net(self, lstm_output, final_state, final_cell_state):
        final_state = final_state.permute([1, 2, 0])
        final_cell_state = final_cell_state.permute([1, 2, 0])
        hidden = torch.cat([final_state, final_cell_state], 1)
        attn_weights = torch.bmm(lstm_output, hidden)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights)
        return new_hidden_state

    def forward(self, input_x):
        # input_x = self.embedding(input_x)
        # input_x = self.embed_dropout(input_x)
        input_x = input_x.unsqueeze(1)
        batch_size = input_x.shape[0]
        h_0 = Variable(torch.zeros(2*LSTM_NUM_LAYERS, batch_size, self.hidden_size))
        c_0 = Variable(torch.zeros(2*LSTM_NUM_LAYERS, batch_size, self.hidden_size))
        output, (final_hidden_state, final_cell_state) = self.lstm(input_x, (h_0, c_0))
        attn_output = self.attention_net(output, final_hidden_state, final_cell_state)
        attn_output = attn_output.view(batch_size, -1)
        logist = self.linear1(attn_output)
        logist = F.relu(logist)
        logist = self.dropout(logist)
        logist = self.linear2(logist)
        out = self.sigmoid(logist)
        return out