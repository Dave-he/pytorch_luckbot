import torch.nn as nn
from config import get_config_int


class TransformerModel(nn.Module):
    def __init__(self):
        input_size = get_config_int('MODEL', 'input_size')
        d_model = get_config_int('MODEL', 'd_model')
        nhead = get_config_int('MODEL', 'nhead')
        num_layers = get_config_int('MODEL', 'num_layers')
        output_size = get_config_int('MODEL', 'output_size')
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, batch_first=True),
            num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]  # 取最后一个时间步的输出
        x = self.fc(x)
        return x
