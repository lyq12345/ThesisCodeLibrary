import pandas as pd
import matplotlib.pyplot as plt
import os

# 设置文件夹路径
folder_path = 'results'

# 初始化总和和计数器
total_power = 0
count = 0

def calculate_mean():
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

def draw_emp_dist(filename):
    df = pd.read_csv(filename)
    data_list = df['proc_time'].to_list()
    sorted_list = sorted(data_list)

    plt.figure(figsize=(6, 4))
    last, i = min(sorted_list), 0
    while i < len(sorted_list):
        plt.plot([last, sorted_list[i]], [i / len(sorted_list), i / len(sorted_list)], 'k')
        if i < len(sorted_list):
            last = sorted_list[i]
        i += 1
    plt.show()

def draw_hist():
    devices = ['pi', 'nano', 'xaiver']
    versions = ['tinyyolov3', 'yolov3']
    fig, axes = plt.subplots(2, 3, sharex='col', sharey='row')
    for i in range(2):
        version_name = versions[i]
        for j in range(3):
            device_name = devices[j]
            file_name = f"results/power_{device_name}_fire_{version_name}.csv"
            df = pd.read_csv(file_name)
            data = df['power'].to_list()
            axes[i, j].hist(data, bins=10, edgecolor='k', color='orange')
            axes[i, j].set_title(f'{version_name} on {device_name}')
    plt.show()

# draw_emp_dist("results/raw_nano_human_tinyyolov3.csv")
draw_hist()


