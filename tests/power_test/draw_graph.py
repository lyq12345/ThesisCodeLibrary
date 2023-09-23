import pandas as pd
import matplotlib as plt
import os

# 设置文件夹路径
folder_path = 'results'

# 初始化总和和计数器
total_power = 0
count = 0

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)

        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 检查是否有"power"列
        if "power" in df.columns:
            # 计算"power"列的平均值并累加到总和中
            power_mean = df["power"].mean()
            print(f"average power for {filename}: {power_mean}")


