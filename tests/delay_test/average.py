import pandas as pd
import matplotlib.pyplot as plt

def draw_avg_dist(filename):
    df = pd.read_csv(filename)
    last_row = df.iloc[-1]

    group1_data = last_row.iloc[[2,3]]
    group2_data = last_row.iloc[[0,1]]

    fig, ax = plt.subplots()
    bar_width = 0.35
    # 绘制第一组数据
    bar1 = ax.bar(range(len(group1_data)), group1_data, width=bar_width, label='Normal')

    # 绘制第二组数据，将位置向右偏移bar_width
    bar2 = ax.bar([i + bar_width for i in range(len(group2_data))], group2_data, width=bar_width, label='GPU')

    # 设置图表标签和标题
    ax.set_xlabel('Operators')
    ax.set_ylabel('power consumption')
    ax.set_title('Human Detection Power Consumption')
    group_names = ['yolov3', 'tinyyolov3']
    ax.set_xticks([i + bar_width / 2 for i in range(len(group_names))])
    ax.set_xticklabels(group_names)  # 假设你的CSV文件有4列数据

    # 显示图例
    ax.legend()
    plt.show()

# draw_emp_dist("results/raw_nano_human_tinyyolov3.csv")
filename = "results/averaged/averaged_power_xavier.csv"
draw_avg_dist(filename)