import pandas as pd

# 读取 CSV 文件
file_path = 'lottery_data.csv'  # 替换为实际的 CSV 文件路径
data = pd.read_csv(file_path)

# # 检查 Date Time 列的数据类型
# if data['Date Time'].dtype != 'datetime64[ns]':
#     # 如果不是日期时间类型，尝试转换
#     try:
#         data['Date Time'] = pd.to_datetime(data['Date Time'], format='%Y%m%d')  # 根据实际格式调整
#     except ValueError:
#         print("日期时间列的格式无法正确转换，请检查格式。")
#         exit()

# 按 Date Time 列进行升序排序
data = data.sort_values(by='Date Time', ascending=True)

# 保存处理后的 CSV 文件
output_file_path = 'lottery_data.csv'  # 替换为你想要保存的文件名
data.to_csv(output_file_path, index=False)

print(f"处理后的文件已保存为 {output_file_path}")