import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
import numpy as np
from scipy.optimize import curve_fit
lable_size = 20
tick_size = 19
legend_size = 18
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
def draw_scatter_and_curve_graph(x_values, y_values, type):
    if type == "cpu" or type == "power":
        popt, pcov = curve_fit(power_model, x_values, y_values, maxfev=10000)
        a_fit, b_fit, c_fit = popt
        residuals = y_values - power_model(x_values, a_fit, b_fit, c_fit)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_values - np.mean(y_values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        print("R-squared:", r_squared)
        if type == "cpu":
            color = "blue"
        else:
            color = "orange"
        # 绘制折线图
        plt.scatter(x_values, y_values, color=color, label="Sampled Data")
        curve_x_values = np.arange(min(x_values), max(x_values)+1)
        plt.plot(curve_x_values, power_model(curve_x_values, a_fit, b_fit, c_fit), color='red', label='Fitted Model')

        # 添加标题和标签
        # plt.title('test')
        if type == "cpu":
            plt.ylabel('CPU Usage (%)', fontsize=lable_size)
            filename = "../figures/cpu_freq.eps"
        else:
            plt.ylabel("Power Consumption Rate (W)", fontsize=lable_size)
            filename = "../figures/power_freq.eps"

        plt.xlabel('Data Arrival Rate', fontsize=lable_size)
        plt.xticks(fontsize=tick_size)
        plt.yticks(fontsize=tick_size)
        # plt.ylim(0, 1000)

        # 显示图例
        plt.legend(fontsize=tick_size)
        plt.tight_layout()

        # 显示折线图

        foo_fig = plt.gcf()  # 'get current figure'
        foo_fig.savefig(filename, format='eps', dpi=2000)
        plt.show()
        # plt.savefig("../figures/cpu_freq.eps", dpi=300)
    else:
        popt, pcov = curve_fit(constant_model, x_values, y_values)
        a_fit = popt
        residuals = y_values - constant_model(x_values, a_fit)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_values - np.mean(y_values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        print("R-squared:", r_squared)
        # 绘制折线图
        plt.scatter(x_values, y_values, color='darkgoldenrod', label="Sampled Data")
        plt.plot(x_values, constant_model(x_values, a_fit), color='red', label='Fitted Curve')

        # 添加标题和标签
        plt.ylabel("Memory Usage (MiB)", fontsize=lable_size)

        plt.xlabel('Data Arrival Rate', fontsize=lable_size)
        plt.ylim(0, 1000)
        plt.xticks(fontsize=tick_size)
        plt.yticks(fontsize=tick_size)

        # 显示图例
        plt.legend(fontsize=tick_size)
        plt.tight_layout()

        # 显示折线图
        # plt.show()
        foo_fig = plt.gcf()  # 'get current figure'
        foo_fig.savefig('../figures/memory_freq.eps', format='eps', dpi=2000)
        plt.show()
        # plt.savefig("../figures/cpu_freq.eps", dpi=300)

def draw_cpu_and_memory(x_values, y_cpu, y_memory):
    popt1, pcov1 = curve_fit(power_model, x_values, y_cpu, maxfev=10000)
    a_fit1, b_fit1, c_fit1 = popt1
    residuals_cpu = y_cpu - power_model(x_values, a_fit1, b_fit1, c_fit1)
    ss_res_cpu= np.sum(residuals_cpu ** 2)
    ss_tot_cpu = np.sum((y_cpu - np.mean(y_cpu)) ** 2)
    r_squared_cpu = 1 - (ss_res_cpu / ss_tot_cpu)
    print("R-squared:", r_squared_cpu)

    popt2, pcov2 = curve_fit(constant_model, x_values, y_memory)
    c_fit2 = popt2
    residuals_memory = y_memory - constant_model(x_values, c_fit2)
    ss_res_memory = np.sum(residuals_memory ** 2)
    ss_tot_memroy = np.sum((y_memory - np.mean(y_memory)) ** 2)
    r_squared_memory = 1 - (ss_res_memory / ss_tot_memroy)
    print("R-squared:", r_squared_memory)
    if type == "cpu":
        color = "blue"
    else:
        color = "orange"

    fig = plt.figure(figsize=(10, 6))

    ax = fig.add_subplot(111)

    ax.scatter(x_values, y_cpu, color='blue', label="CPU")
    curve_x_values = np.arange(min(x_values), max(x_values) + 1)
    ax.plot(curve_x_values, power_model(curve_x_values, a_fit1, b_fit1, c_fit1), color='red')
    ax.set_xlabel("Data arrival rate", fontsize=lable_size)
    ax.set_ylabel("CPU Usage (%)", fontsize=lable_size)


    ax2 = ax.twinx()
    ax2.scatter(x_values, y_memory, color='darkgoldenrod', label='Memory')
    ax2.plot(x_values, constant_model(x_values, c_fit2), color='red')

    fig.legend(loc=1, bbox_to_anchor=(1, 0.2), bbox_transform=ax.transAxes)


    ax2.set_ylabel(r"Memory Usage (MB)", fontsize=lable_size)
    ax2.set_ylim(0, 1000)
    ax2.set_xlabel("data arrival rate", fontsize=lable_size)
    ax.set_ylabel(r"CPU Usage (%)", fontsize=lable_size)

    # 调整所有坐标轴刻度的字体大小
    ax.tick_params(axis='x', labelsize=tick_size)
    ax.tick_params(axis='y', labelsize=tick_size)
    ax2.tick_params(axis='y', labelsize=tick_size)



    plt.savefig('cpu_and_memory.png')



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
data1 = pd.read_csv('../../../power_test/results/power/evaluation.csv')
data2 = pd.read_csv('../qos_tests/evaluation.csv')
# y = data['qos']
x1 = data2['cpu']
x2 = data2['memory']
x3 = data1['power']
x3 = [x*0.001 for x in x3]
x4 = data2['interval']
x4_freq = [1/item for item in x4]
# print(x4_freq)
# draw_scatter_graph(x4_freq, x1, "cpu")
# calculate_p_value(x1, y)
# calculate_p_value(x2, y)
# calculate_standard_deviation(y)

# def func(x, a, b, c):
#     return a * np.exp(-b * x) + c
# def log_model(x, a, b):
#     return a * np.log(x) + b
#
def constant_model(x, a):
    return np.full_like(x, a)
def power_model(x, a, b, c):
    return a * np.power(x, b) + c

# print(power_model(1.2, a_fit, b_fit, c_fit))

# 绘制原始数据和拟合曲线
draw_scatter_and_curve_graph(x4_freq, x1, "cpu")
# draw_scatter_and_curve_graph(x4_freq, x2, "memory")
draw_scatter_and_curve_graph(x4_freq, x3, "power")
# draw_cpu_and_memory(x4_freq, x1, x2)