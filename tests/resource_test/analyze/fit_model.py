import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
import numpy as np
def draw_scatter_graph(x_values, y_values):

    # 绘制折线图
    plt.scatter(x_values, y_values, color='orange')

    # 添加标题和标签
    # plt.title('test')
    plt.xlabel('invocation frequency(times/s)')
    plt.ylabel('memory_usage(MiB)')
    plt.ylim(0, 1000)

    # 显示图例
    plt.legend()

    # 显示折线图
    plt.show()

def calculate_p_value(x, y):
    correlation_coefficient, p_value = pearsonr(x, y)

    # 打印结果
    print(f"Pearson相关系数: {correlation_coefficient}")
    print(f"P-value: {p_value}")

    # 判断是否显著
    alpha = 0.05  # 设置显著性水平
    if p_value < alpha:
        print("相关性显著")
    else:
        print("相关性不显著")

def calculate_standard_deviation(data):
    std_deviation = np.std(data)

    # 打印结果
    print(f"标准差: {std_deviation}")

    # 设置阈值判断波动
    threshold = 1.0  # 根据实际情况调整阈值
    if std_deviation < threshold:
        print("变量在某个水平上下波动")
    else:
        print("变量波动较大")
data = pd.read_csv('../qos_tests/evaluation_2.csv')
y = data['qos']
x1 = data['cpu']
x2 = data['memory']
x4 = data['interval']
x4_freq = [1/item for item in x4]
print(x4_freq)
draw_scatter_graph(x4_freq, x2)
# calculate_p_value(x1, y)
# calculate_p_value(x2, y)
# calculate_standard_deviation(y)