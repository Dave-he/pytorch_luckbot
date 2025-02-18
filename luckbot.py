import torch
import numpy as np
import matplotlib.pyplot as plt

from nets.lstmNet import LSTMModel
from train import train_model
from dataloader import load_data
from config import *


# 加载配置文件
load_config()

# 1. 数据准备


def load_and_preprocess_data(file_path):
    scaled_data, scaler = load_data(file_path)

    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    def create_sequences(data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data[i:i + seq_length]
            y = data[i + seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    seq_length = get_config_int('DATA', 'seq_length')
    X_train, y_train = create_sequences(train_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)

    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()

    return X_train, y_train, X_test, y_test, scaler


# 3. 训练模型
def train(model, X_train, y_train):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = get_config_int('TRAINING', 'batch_size')
    lr = get_config_float('TRAINING', 'learning_rate')
    X_train = X_train.to(device)
    y_train = y_train.to(device)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    criterion = torch.nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.1)

    epochs = get_config_int('TRAINING', 'epochs')
    patience = get_config_int('TRAINING', 'patience')
    return train_model(model, train_loader, criterion,
                       optimizer, scheduler, epochs, patience)


# 4. 模型评估和预测
def evaluate_model(model, X_test, y_test, scaler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        criterion = torch.nn.MSELoss()
        test_loss = criterion(test_outputs, y_test)
        print(f'Test Loss: {test_loss.item():.4f}')

        test_outputs = test_outputs.cpu().numpy()
        y_test = y_test.cpu().numpy()
        test_outputs = scaler.inverse_transform(test_outputs)
        y_test = scaler.inverse_transform(y_test)

        plt.plot(y_test[:, 0], label='Actual')
        plt.plot(test_outputs[:, 0], label='Predicted')
        plt.xlabel('Time')
        plt.ylabel('Lottery Number')
        plt.legend()
        plt.show()


def main(file_path):
    X_train, y_train, X_test, y_test, scaler = load_and_preprocess_data(
        file_path)

    model = LSTMModel()
    trained_model = train(model, X_train, y_train)
    # 保存模型
    model_path = f'models/lottery_{get_config('MODEL', 'lstm_key')}.pth'
    torch.save(trained_model.state_dict(), model_path)
    print(f"模型已保存到 {model_path}")

    evaluate_model(trained_model, X_test, y_test, scaler)


if __name__ == "__main__":
    main('lottery_data.csv')
