import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(2, 5)  # 输入层到隐藏层
        self.fc2 = nn.Linear(5, 1)  # 隐藏层到输出层
 
    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
 