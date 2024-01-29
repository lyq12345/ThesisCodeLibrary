import os

import numpy as np

def remove_outliers_percentile(data, lower_percentile=5, upper_percentile=95):
    # 计算百分位数
    lower_limit = np.percentile(data, lower_percentile)
    upper_limit = np.percentile(data, upper_percentile)

    # 剔除位于百分位数之外的值
    filtered_data = [value for value in data if lower_limit <= value <= upper_limit]

    return filtered_data

def process_log_file(input_log_path, output_log_path):
    # 读取.log文件
    with open(input_log_path, 'r') as file:
        lines = file.readlines()

    # 提取第二列数据
    cpu_data = [float(line.split('   ')[1][:-1]) for line in lines]
    memory_data = [float(line.split('   ')[2].split("MiB")[0]) for line in lines]
    # cpu_data = [float(line.split('   ')[1][:-1]) for line in lines]

    # 使用百分位法剔除离群值
    filtered_cpu = remove_outliers_percentile(cpu_data)
    filtered_memory = remove_outliers_percentile(memory_data)

    avg_cpu = sum(filtered_cpu) / len(filtered_cpu)
    avg_memory = sum(filtered_memory) / len(filtered_memory)

    print(f"Average CPU: {avg_cpu}")
    print(f"Average Memory: {avg_memory}")


    #
    # # 将结果写回.log文件
    # with open(output_log_path, 'w') as output_file:
    #     for line, value in zip(lines, filtered_data):
    #         output_file.write(f"{line.split('\t')[0]}\t{value}\t{line.split('\t')[2]}")

# 示例使用
input_log_path = '../results/docker_stats_1.log'
output_log_path = '../qos_tests/docker_stats_01_filtered.log'

input_folder = '../results'
file_list = []
for root, dirs, files in os.walk(input_folder):
    for file in files:
        file_list.append(os.path.join(root, file))

for file in file_list:
    print(file)
    process_log_file(file, output_log_path)
