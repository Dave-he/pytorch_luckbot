import torch
import numpy as np
from dataloader import load_data
from nets import TransformerModel
from train import train_model, save_model
from config import load_config, get_config_int, get_config

load_config()

# 3. 定义红球不重复的物理信息损失
def non_repeat_loss(output):
    red_balls = output[:, :6]  # 假设前 6 个是红球
    batch_size = red_balls.size(0)
    loss = 0
    for i in range(batch_size):
        for j in range(6):
            for k in range(j + 1, 6):
                loss += (red_balls[i, j] - red_balls[i, k]) ** 2
    return loss / batch_size

# 4. 定义总损失函数
def total_loss(model_output, target):
    mse_loss = torch.nn.MSELoss()
    data_loss = mse_loss(model_output, target)
    physics_loss = non_repeat_loss(model_output)
    total = data_loss + physics_loss
    return total

# 5. 准备数据加载器
def create_data_loader(features, seq_length, batch_size):
    inputs = []
    targets = []
    for i in range(len(features) - seq_length):
        inputs.append(features[i:i+seq_length])
        targets.append(features[i+seq_length])
    
    # 先将列表转换为单个 numpy.ndarray
    inputs = np.array(inputs)
    targets = np.array(targets)
    
    # 再将 numpy.ndarray 转换为 PyTorch 张量
    inputs = torch.tensor(inputs, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)
    
    dataset = torch.utils.data.TensorDataset(inputs, targets)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader

# 7. 处理预测结果，使其成为符合彩票规则的整数
def process_prediction(prediction, red_min=1, red_max=33, blue_min=1, blue_max=16):
    # 四舍五入取整
    prediction_rounded = np.round(prediction).astype(int)
    # 处理红球
    red_balls = prediction_rounded[0, :6]
    red_balls = np.clip(red_balls, red_min, red_max)
    unique_red_balls = np.unique(red_balls)
    while len(unique_red_balls) < 6:
        new_ball = np.random.randint(red_min, red_max + 1)
        while new_ball in unique_red_balls:
            new_ball = np.random.randint(red_min, red_max + 1)
        unique_red_balls = np.append(unique_red_balls, new_ball)
    # 处理蓝球
    blue_ball = prediction_rounded[0, 6]
    blue_ball = np.clip(blue_ball, blue_min, blue_max)
    final_prediction = np.concatenate((unique_red_balls, [blue_ball]))
    return final_prediction

# 8. 模型预测
def predict(model, features, seq_length, scaler):
    last_sequence = features[-seq_length:]
    last_sequence = np.expand_dims(last_sequence, axis=0)
    last_sequence = torch.tensor(last_sequence, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        prediction = model(last_sequence)
    prediction = prediction.numpy()
    prediction = scaler.inverse_transform(prediction)
    processed_prediction = process_prediction(prediction)
    return processed_prediction


# 主函数
def main():
    file_path = 'lottery_data.csv'
    features, scaler = load_data(file_path)
    model = TransformerModel()

    criterion = total_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    seq_length = get_config_int('DATA', 'seq_length')
    batch_size = get_config_int('TRAINING', 'batch_size')
    epochs = get_config_int('TRAINING', 'epochs')
    
    train_loader = create_data_loader(features, seq_length, batch_size)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    trained_model = train_model(model, train_loader, criterion, optimizer, scheduler, epochs)

    save_model(trained_model, get_config('MODEL', 'trans_key'))

    prediction = predict(model, features, seq_length, scaler)
    print("预测的彩票号码：", prediction)

if __name__ == "__main__":
    main()