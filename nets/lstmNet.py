import torch
from config import get_config_int


# 2. 模型构建
class LSTMModel(torch.nn.Module):

    def __init__(self):
        super(LSTMModel, self).__init__()
        input_size = get_config_int('MODEL', 'input_size')
        hidden_size = get_config_int('MODEL', 'hidden_size')
        num_layers = get_config_int('MODEL', 'num_layers')
        output_size = get_config_int('MODEL', 'output_size')
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)
        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                torch.nn.init.kaiming_normal_(param)
            elif 'bias' in name:
                torch.nn.init.zeros_(param)
        torch.nn.init.kaiming_normal_(self.fc.weight)
        torch.nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out