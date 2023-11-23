import json
import time
import os
import numpy as np
import sys
import argparse
import pandas as pd
import random
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
# from greedy_deploy import Greedy_Decider
from MIP_deploy import MIP_Decider
from TOPSIS_deploy import TOPSIS_decider
from LocalSearch_deploy import LocalSearch_deploy
from ORTools_deploy import ORTools_Decider
from examples.testcases import generate_testcase

from status_tracker.task_mock import generate_tasks
from status_tracker.device_mock import generate_devices

cur_dir = os.getcwd()

speed_lookup_table = {
  0: {
    "jetson-nano": 0.5549,
    "raspberrypi-4b": 1.0702,
    "jetson-xavier": 0.4276
  },
  1: {
        "jetson-nano": 4.364,
        "raspberrypi-4b": 7.0823,
        "jetson-xavier": 2.6235
    },
  2: {
    "jetson-nano": 0.5864,
    "raspberrypi-4b": 1.0913,
    "jetson-xavier": 0.4605
  },
  3: {
    "jetson-nano": 4.4829,
    "raspberrypi-4b": 7.2191,
    "jetson-xavier": 3.8648
  }
}

data = {'group':[], 'Objective':[],'Normalized objective':[], 'time':[], 'algorithm': [], "avg_accuracy": [], "avg_delay": [], "avg_cpu_consumption": [], "avg_memory_consumption": [],"power_consumption": []}

power_lookup_table = {
  "joelee0515/firedetection:yolov3-measure-time": {
    "jetson-nano": 2916.43,
    "raspberrypi-4b": 1684.4,
    "jetson-xavier": 1523.94
  },
  "joelee0515/firedetection:tinyyolov3-measure-time": {
    "jetson-nano": 1584.53,
    "raspberrypi-4b": 1174.39,
    "jetson-xavier": 780.97
  },
  "joelee0515/humandetection:yolov3-measure-time": {
    "jetson-nano": 2900.08,
    "raspberrypi-4b": 1694.41,
    "jetson-xavier": 1540.61
  },
  "joelee0515/humandetection:tinyyolov3-measure-time": {
    "jetson-nano": 1191.19,
    "raspberrypi-4b": 1168.31,
    "jetson-xavier": 803.95
  }
}

def read_json(filename):
    with open(filename, 'r') as json_file:
        data = json.load(json_file)
    return data



def choose_best_operator(operator_candidates):
    max_speed_op = max(operator_candidates, key=lambda x: x["speed"])
    return max_speed_op


def generate_transmission_rate_matrix(n, min_rate=1, max_rate=5):
    transmission_matrix = np.full((n, n), np.inf)

    # diagnose to zero
    np.fill_diagonal(transmission_matrix, 0)

    # get random link rates between pairs
    for i in range(n):
        for j in range(i + 1, n):
            rate = random.uniform(min_rate, max_rate)
            transmission_matrix[i, j] = rate
            transmission_matrix[j, i] = rate  # 对称性

    return transmission_matrix

def make_decision_from_task_new(task_list, device_list, transmission_matrix, solver="LocalSearch", display=True, record=False):
    operator_file = os.path.join(cur_dir, "../status_tracker/operators.json")
    operator_list = read_json(operator_file)

    def calculate_accuracy(operator_id):
        return operator_list[operator_id]["accuracy"]

    def calculate_delay(operator_id, source_device_id, device_id):
        device_model = device_list[device_id]["model"]
        transmission_delay = transmission_matrix[source_device_id, device_id]
        processing_delay = speed_lookup_table[operator_id][device_model]
        return transmission_delay + processing_delay

    def calculate_objective(op_id, source_device_id, device_id, task_delay):
        acc = calculate_accuracy(op_id)
        delay = calculate_delay(op_id, source_device_id, dev_id)
        objective = acc - max(0, (delay - task_delay)/delay)
        return objective


    def calculate_power(operator_id, device_id):
        operator_name = operator_list[operator_id]["name"]
        device_model = device_list[device_id]["model"]
        power = power_lookup_table[operator_name][device_model]
        return power

    def calculate_resource_consumption(solution):
        cpu_consumptions = [0]*len(device_list)
        ram_consumptions = [0]*len(device_list)
        for i, mapping in enumerate(solution):
            op_id = mapping[0]
            dev_id = mapping[1]
            op_resource = operator_list[op_id]["requirements"]["system"]
            cpu_consumptions[dev_id] += op_resource["cpu"]
            ram_consumptions[dev_id] += op_resource["memory"]
        for i in range(len(cpu_consumptions)):
            cpu_consumptions[i] = cpu_consumptions[i] / device_list[i]["resources"]["system"]["cpu"]
            ram_consumptions[i] = ram_consumptions[i] / device_list[i]["resources"]["system"]["memory"]
        avg_cpu_consumption = sum(cpu_consumptions)/len(cpu_consumptions)
        avg_ram_consumption = sum(ram_consumptions)/len(ram_consumptions)
        return avg_cpu_consumption, avg_ram_consumption



    # decision_maker = MIP_Decider(task_list, device_list, operator_list)
    decision_maker = None
    if solver == "TOPSIS":
        decision_maker = TOPSIS_decider(task_list, device_list, operator_list, transmission_matrix)
    elif solver == "LocalSearch":
        decision_maker = LocalSearch_deploy(task_list, device_list, operator_list, transmission_matrix)
    elif solver == "MIP":
        decision_maker = MIP_Decider(task_list, device_list, operator_list, transmission_matrix)
    elif solver == "ORTools":
        decision_maker = ORTools_Decider(task_list, device_list, operator_list, transmission_matrix)
    start_time = time.time()
    solution, utility = decision_maker.make_decision()
    end_time = time.time()
    elapsed_time = end_time - start_time
    if display:
        print("Solution: ")
        for i, mapping in enumerate(solution):
            task_object = task_list[i]['object']
            task_delay = task_list[i]['delay']
            source_device_id = task_list[i]["source"]
            op_id = mapping[0]
            op_name = operator_list[op_id]["name"]
            dev_id = mapping[1]
            performance_acc = calculate_accuracy(op_id)
            performance_delay = calculate_delay(op_id, source_device_id, dev_id)
            performance_power = calculate_power(op_id, dev_id)
            performance_objective = calculate_objective(op_id, source_device_id, dev_id, task_delay)
            print(f"Data flow {i}:")
            print(f"Request: source: {source_device_id} object: {task_object}, delay tolerance: {task_delay}")
            print(f"Deployment: {op_name} -> device {dev_id}")
            print(f"Performance: accuracy: {performance_acc}, delay: {performance_delay}, power: {performance_power}, objective: {performance_objective}")
            print("--------------------------------------------------------------")
        print(f"Decision making time: {elapsed_time} s")
        print(f"Overall Objective: {utility}")
    if record:
        nol_objective = utility / len(task_list)
        # data['group'].append(f"i={len(task_list)} k={len(device_list)}")
        data['group'].append(len(task_list))
        data['Objective'].append(utility)
        data['Normalized objective'].append(nol_objective)
        data['time'].append(elapsed_time)
        data['algorithm'].append(solver)
        acc_sum = 0.0
        delay_sum = 0.0
        power_sum = 0.0

        for i, mapping in enumerate(solution):
            source_device_id = task_list[i]["source"]
            op_id = mapping[0]
            dev_id = mapping[1]
            acc_sum += calculate_accuracy(op_id)
            delay_sum += calculate_delay(op_id, source_device_id, dev_id)
            power_sum += calculate_power(op_id, dev_id)
        avg_acc = acc_sum / len(task_list)
        avg_delay = delay_sum / len(task_list)
        avg_cpu_consumption, avg_ram_consumption = calculate_resource_consumption(solution)
        data["avg_accuracy"].append(avg_acc)
        data["avg_delay"].append(avg_delay)
        data["power_consumption"].append(power_sum)
        data["avg_cpu_consumption"].append(avg_cpu_consumption)
        data["avg_memory_consumption"].append(avg_ram_consumption)

def main():
    num_devices = 20
    num_requests = 6
    solver = "LocalSearch"

    # 创建一个 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='示例脚本，演示如何使用 argparse 解析命令行参数.')

    # 添加命令行参数
    parser.add_argument('-d', '--num_devices', default=30, type=int, help='number of devices')
    parser.add_argument('-r', '--num_requests', default=10, type=float, help='number of requests')
    parser.add_argument('-s', '--solver', type=str, default='LocalSearch', help='solver name')

    # 解析命令行参数
    args = parser.parse_args()

    # 访问解析后的参数
    num_devices = args.num_devices
    num_requests = args.num_requests
    solver = args.solver

    device_list = generate_devices(num_devices)
    task_list = generate_tasks(num_requests, device_list)
    transmission_matrix = generate_transmission_rate_matrix(len(device_list))
    # device_list, task_list, transmission_matrix = generate_testcase()


    if solver == "All":
        make_decision_from_task_new(task_list, device_list, transmission_matrix, "LocalSearch")
        make_decision_from_task_new(task_list, device_list, transmission_matrix, "ORTools")
    else:
        make_decision_from_task_new(task_list, device_list, transmission_matrix, solver)
    # make_decision_from_task_new(task_list, device_list, transmission_matrix, "TOPSIS")
    # make_decision_from_task_new(task_list, device_list, transmission_matrix, "MIP")

def evaluation_experiments():
    # num_devices = [5, 10, 20, 30, 40, 50, 100]
    # num_requests = [5, 10, 20, 30, 40, 50, 100]
    num_devices = [20]
    num_requests = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    measure_times = 1
    solvers = ["TOPSIS", "LocalSearch", "MIP"]

    for i, device_num in enumerate(num_devices):
        # for j in range(i + 1):
        for j in range(len(num_requests)):
            for t in range(measure_times):
                request_num = num_requests[j]
                device_list = generate_devices(device_num)
                task_list = generate_tasks(request_num, device_list)
                transmission_matrix = generate_transmission_rate_matrix(len(device_list))
                for solver in solvers:
                    if request_num > 6 and solver == "MIP":
                        continue
                    print(f"Running i={request_num} k={device_num}, solver={solver}, iteration {t}")
                    make_decision_from_task_new(task_list, device_list, transmission_matrix, solver, display=False, record=True)

    # record finishes, save into csv
    df = pd.DataFrame(data)
    df.to_csv('results/evaluation_4.csv', index=False)
if __name__ == '__main__':
    main()
    # evaluation_experiments()

