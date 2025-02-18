import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#1. 数据加载与预处理
def load_data(file_path):
    data = pd.read_csv(file_path)

    # 提取彩票号码特征
    features = data[['red1', 'red2', 'red3', 'red4', 'red5', 'red6', 'blue']].values
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)
    
    return features, scaler