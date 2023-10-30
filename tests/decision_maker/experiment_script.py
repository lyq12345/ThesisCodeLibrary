import subprocess

num_devices = [5, 10, 20, 30, 40, 50, 100]
num_tasks = [5, 10, 20, 30, 40, 50, 100]
measure_times = 50
solvers = ["TOPSIS", "LocalSearch"]
# 循环遍历1到20的数
for i, device_num in enumerate(num_devices):
    for j in range(i+1):
        task_num = num_tasks[j]
