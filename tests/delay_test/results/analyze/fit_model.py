import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
import numpy as np
from scipy.optimize import curve_fit
def draw_scatter_graph(x_values, y_values):

    # 绘制折线图
    plt.scatter(x_values, y_values, color='orange')

    # 添加标题和标签
    # plt.title('test')
    plt.xlabel('invocation frequency(times/s)')
    # plt.ylabel('memory_usage(MiB)')
    plt.ylabel('cpu_load(%)')
    # plt.ylim(0, 1000)

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
# draw_scatter_graph(x4_freq, x1)
# calculate_p_value(x1, y)
# calculate_p_value(x2, y)
# calculate_standard_deviation(y)

# def func(x, a, b, c):
#     return a * np.exp(-b * x) + c
def log_model(x, a, b):
    return a * np.log(x) + b

def power_model(x, a, b, c):
    return a * np.power(x, b) + c

popt, pcov = curve_fit(power_model, x4_freq, x1, maxfev=10000)

a_fit, b_fit, c_fit = popt
print(a_fit, b_fit, c_fit)

# 评估拟合效果
residuals = x1 - power_model(x4_freq, a_fit, b_fit, c_fit)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((x1 - np.mean(x1))**2)
r_squared = 1 - (ss_res / ss_tot)
print("R-squared:", r_squared)

print(power_model(1.2, a_fit, b_fit, c_fit))

# 绘制原始数据和拟合曲线
plt.scatter(x4_freq, x1, label='Original Data')
plt.plot(x4_freq, power_model(x4_freq, a_fit, b_fit, c_fit), color='red', label='Fitted Curve')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Power Model Fitting')
plt.show()