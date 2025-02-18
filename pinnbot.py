import torch

from nets import LSTMModel
from train import train_model, save_model
from dataloader import load_data
from config import load_config, get_config, get_config_int

# 加载配置文
load_config()

# 3. 定义红球不重复的物理信息损失
def non_repeat_loss(output):
    red_balls = output[:, :6]  # 假设前 6 个是红球
    batch_size = red_balls.size(0)
    loss = 0
    for i in range(batch_size):
        for j in range(6):
            for k in range(j + 1, 6):
                # 计算红球之间的差值平方和作为损失
                loss += (red_balls[i, j] - red_balls[i, k]) ** 2
    return loss / batch_size

# 4. 定义总损失函数
def total_loss(lstm_output, target):
    mse_loss = torch.nn.MSELoss()
    data_loss = mse_loss(lstm_output, target)
    physics_loss = non_repeat_loss(lstm_output)
    # 可以调整权重来平衡数据损失和物理信息损失
    total = data_loss + physics_loss
    return total

# 5. 准备数据加载器
def create_data_loader(file_path):
    features, _ = load_data(file_path)
    batch_size = get_config_int('TRAINING', 'batch_size')
    seq_length = get_config_int('DATA', 'seq_length')
    inputs = []
    targets = []
    for i in range(len(features) - seq_length):
        inputs.append(features[i:i+seq_length])
        targets.append(features[i+seq_length])
    # 明确指定数据类型为 torch.float32
    inputs = torch.tensor(inputs, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(inputs, targets)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True)
    return data_loader

# 主函数
def main(file_path):
    train_loader = create_data_loader(file_path)
    model = LSTMModel()
    criterion = total_loss
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.1)
    trained_model = train_model(
        model, train_loader, criterion, optimizer, scheduler)

    # 保存模型
    key = get_config('MODEL', 'pinn_key')
    save_model(trained_model, key)


if __name__ == "__main__":
    main('lottery_data.csv')
