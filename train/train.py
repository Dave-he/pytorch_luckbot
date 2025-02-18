from datetime import datetime
import torch

# 6. 模型训练


def train_model(model, train_loader, criterion, optimizer, scheduler, epochs=100, patience=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    best_loss = float('inf')
    counter = 0
    for epoch in range(epochs):
        total_loss_value = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss_value += loss.item()

        epoch_loss = total_loss_value / len(train_loader)
        scheduler.step()

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping!")
                break

        print(
            f'{datetime.now()} Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}')

    return model


def save_model(model, key='LSTM'):
    model_path = f'models/lottery_{key}.pth'
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到 {model_path}")