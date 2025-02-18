import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from nets.lstmNet import LSTMModel
from config import get_config, load_config

load_config()

# 根据输入时间进行单点预测函数
def predict_single_point_by_time(model, input_time, data, scaler, seq_length=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 检查输入时间格式
    if not isinstance(input_time, str) or not input_time.isdigit():
        print("输入的彩票期数格式不正确，请输入如 '2025016' 这样的数字字符串。")
        return

    # 检查输入时间是否存在于数据中
    if input_time not in data['Date Time'].astype(str).values:
        print(f"输入期数 {input_time} 不存在于数据中，请检查输入。")
        return

    # 找到输入时间对应的索引
    time_index = data[data['Date Time'].astype(str) == input_time].index[0]

    # 检查序列长度是否足够
    if time_index < seq_length:
        print("输入期数对应的序列长度不足，无法进行预测。")
        return

    # 提取输入序列
    input_sequence = data.loc[time_index - seq_length:time_index - 1, ['red1', 'red2', 'red3', 'red4', 'red5', 'red6', 'blue']]
 
    # 对输入序列进行归一化
    input_sequence = pd.DataFrame(scaler.transform(input_sequence), columns=input_sequence.columns)
 
    # 将输入序列转换为PyTorch张量并调整维度
    input_tensor = torch.from_numpy(input_sequence.values).float().unsqueeze(0).to(device)
 
    model.eval()
    with torch.no_grad():
        prediction = model(input_tensor)

    # 将预测结果转换为 numpy 数组并反归一化
    prediction = prediction.cpu().numpy()
    prediction = scaler.inverse_transform(prediction)

    # 输出 7 个号码
    red_nums = sorted([int(num) for num in prediction[0][:6]])
    blue_num = int(prediction[0][6])

    return red_nums + [blue_num]

def main(file_path):
    # 预处理数据
    data = pd.read_csv(file_path)
    if data is None:
        exit()

    features = data[['red1', 'red2', 'red3', 'red4', 'red5', 'red6', 'blue']]
    scaler = MinMaxScaler()
    scaler.fit_transform(features)

    model = LSTMModel()
    model_path = f'models/lottery_{get_config('MODEL', 'pinn_key')}.pth'
    model.load_state_dict(torch.load(model_path))

    # 输入时间
    input_time = '2025016'  # 替换为实际的输入时间
    prediction = predict_single_point_by_time(model, input_time, data, scaler)
    if prediction:
        print(f"输入时间 {input_time} 的单点预测结果（6 个红球 + 1 个蓝球）: {prediction}")

if __name__ == "__main__":
   main('lottery_data.csv')