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
# from ORTools_deploy import ORTools_Decider
from new.ORTools_deploy import ORTools_Decider
from Greedy_deploy import Greedy_decider
from LocalSearch_new import LocalSearch_new

# from status_tracker.task_mock import generate_tasks
from status_tracker.workflow_mock import generate_workflows
from status_tracker.device_mock import generate_devices

cur_dir = os.getcwd()

speed_lookup_table = None
power_lookup_table = None
with open(os.path.join(cur_dir, "../status_tracker/speed_lookup_table.json"), 'r') as file:
    speed_lookup_table = json.load(file)
with open(os.path.join(cur_dir, "../status_tracker/power_lookup_table.json"), 'r') as file:
    power_lookup_table = json.load(file)

data = {'group': [], 'Normalized objective': [], 'time': [], 'algorithm': []}

def read_json(filename):
    with open(filename, 'r') as json_file:
        data = json.load(json_file)
    return data

def print_table(data):
    # 打印表头
    print("|", end="")
    for header in data[0]:
        print(f" {header} |", end="")
    print()

    # 打印分隔线
    print("|", end="")
    for _ in data[0]:
        print("------|", end="")
    print()

    # 打印数据行
    for row in data[1:]:
        print("|", end="")
        for item in row:
            print(f" {item} |", end="")
        print()

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
            transmission_matrix[j, i] = rate  # (i,j) = (j,i)

    return transmission_matrix


def make_decision_from_task_new(workflow_list, device_list, transmission_matrix, solver="LocalSearch", display=True,
                                record=False, iterations=1):
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
        objective = acc - max(0, (delay - task_delay) / delay)
        return objective

    def calculate_power(operator_id, device_id):
        # operator_name = operator_list[operator_id]["name"]
        device_model = device_list[device_id]["model"]
        power = power_lookup_table[operator_id][device_model]
        return power

    def calculate_resource_consumption(solution):
        cpu_consumptions = [0] * len(device_list)
        ram_consumptions = [0] * len(device_list)
        for i, mapping in enumerate(solution):
            op_id = mapping[0]
            dev_id = mapping[1]
            op_resource = operator_list[op_id]["requirements"]["system"]
            cpu_consumptions[dev_id] += op_resource["cpu"]
            ram_consumptions[dev_id] += op_resource["memory"]
        for i in range(len(cpu_consumptions)):
            cpu_consumptions[i] = cpu_consumptions[i] / device_list[i]["resources"]["system"]["cpu"]
            ram_consumptions[i] = ram_consumptions[i] / device_list[i]["resources"]["system"]["memory"]
        print("CPU consumptions: ")
        print(cpu_consumptions)
        print("Memory consumptions:")
        print(ram_consumptions)
        avg_cpu_consumption = sum(cpu_consumptions) / len(cpu_consumptions)
        avg_ram_consumption = sum(ram_consumptions) / len(ram_consumptions)
        return avg_cpu_consumption, avg_ram_consumption

    decision_maker = None
    if solver == "TOPSIS":
        decision_maker = TOPSIS_decider(workflow_list, device_list, operator_list, transmission_matrix)
    elif solver == "LocalSearch":
        decision_maker = LocalSearch_deploy(workflow_list, device_list, operator_list, transmission_matrix)
    elif solver == "ORTools":
        decision_maker = ORTools_Decider(workflow_list, device_list, operator_list, transmission_matrix)
    elif solver == "Greedy":
        decision_maker = Greedy_decider(workflow_list, device_list, operator_list, transmission_matrix)
    elif solver == "LocalSearch_new":
        decision_maker = LocalSearch_new(workflow_list, device_list, operator_list, transmission_matrix)

    sum_elapsed_time = 0.0
    sum_utility = 0.0

    res_objective = 0
    res_time = 0

    # for i in range(iterations):
    #     print(f"Running iteration {i + 1} ...")
    #
    #     start_time = time.time()
    #     solution, utility = decision_maker.make_decision()
    #     res_objective = utility
    #     end_time = time.time()
    #     elapsed_time = end_time - start_time
    #     res_time = elapsed_time
    #
    #     sum_elapsed_time += elapsed_time
    #     sum_utility += utility
    #
    #     if display:
    #         print("Solution: ")
    #         for i, mapping in enumerate(solution):
    #             task_object = workflow_list[i]['object']
    #             task_delay = workflow_list[i]['delay']
    #             source_device_id = workflow_list[i]["source"]
    #             op_id = mapping[0]
    #             op_name = operator_list[op_id]["name"]
    #             dev_id = mapping[1]
    #             performance_acc = calculate_accuracy(op_id)
    #             performance_delay = calculate_delay(op_id, source_device_id, dev_id)
    #             # if performance_delay > task_delay:
    #             #     delay_deviation_sum += round((performance_delay-task_delay)/task_delay, 2)
    #             performance_power = calculate_power(op_id, dev_id)
    #             performance_objective = calculate_objective(op_id, source_device_id, dev_id, task_delay)
    #             print(f"Data flow {i}:")
    #             print(f"Request: source: {source_device_id} object: {task_object}, delay tolerance: {task_delay}")
    #             print(f"Deployment: {op_name} -> device {dev_id}")
    #             print(
    #                 f"Performance: accuracy: {performance_acc}, delay: {performance_delay}, power: {performance_power}, objective: {performance_objective}")
    #             print("--------------------------------------------------------------")
    #         print(f"Decision making time: {elapsed_time} s")
    #         calculate_resource_consumption(solution)
    #         # print(f"Accuracy deviation: {acc_deviation_sum / len(task_list)}")
    #         # print(f"Delay deviation: {delay_deviation_sum / len(task_list)}")
    #         print(f"Overall Objective: {utility}")
    #
    # if record:
    #     avg_nol_objective = (sum_utility / iterations) / len(workflow_list)
    #     avg_time = sum_elapsed_time / iterations
    #
    #     data['Normalized objective'].append(avg_nol_objective)
    #     data['time'].append(avg_time)
    #     data['group'].append(len(workflow_list))
    #     data['algorithm'].append(solver)
    return res_objective, res_time


def main():
    num_devices = 20
    num_requests = 6
    solver = "LocalSearch"

    parser = argparse.ArgumentParser(description='test script.')

    parser.add_argument('-d', '--num_devices', default=50, type=int, help='number of devices')
    parser.add_argument('-r', '--num_requests', default=20, type=float, help='number of requests')
    parser.add_argument('-s', '--solver', type=str, default='ORTools', help='solver name')

    args = parser.parse_args()

    num_devices = args.num_devices
    num_requests = args.num_requests
    solver = args.solver

    device_list = generate_devices(num_devices)
    workflow_list = generate_workflows(num_requests, device_list)
    # print(workflow_list)
    # task_list = generate_tasks(num_requests, device_list)
    transmission_matrix = generate_transmission_rate_matrix(len(device_list))

    table_data = [
        ["Algorithm", "Objective", "Time"]
    ]

    if solver == "All":
        obj_1, time_1 = make_decision_from_task_new(workflow_list, device_list, transmission_matrix, "LocalSearch")
        table_data.append(["LocalSearch", obj_1, time_1])

        obj_2, time_2 = make_decision_from_task_new(workflow_list, device_list, transmission_matrix, "LocalSearch_new")
        table_data.append(["LocalSearch_new", obj_2, time_2])

        obj_3, time_3 = make_decision_from_task_new(workflow_list, device_list, transmission_matrix, "TOPSIS")
        table_data.append(["TOPSIS", obj_3, time_3])

        obj_4, time_4 = make_decision_from_task_new(workflow_list, device_list, transmission_matrix, "ORTools")
        table_data.append(["ORTools", obj_4, time_4])

        print("Summary:")
        print_table(table_data)

    else:
        obj, time = make_decision_from_task_new(workflow_list, device_list, transmission_matrix, solver)
        table_data.append([solver, obj, time])
    # make_decision_from_task_new(task_list, device_list, transmission_matrix, "TOPSIS")
    # make_decision_from_task_new(task_list, device_list, transmission_matrix, "MIP")


def evaluation_experiments():
    num_devices = [100]
    num_requests = [i for i in range(1, 21)]
    # num_devices = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    # num_requests = [10]
    measure_times = 1
    solvers = ["LocalSearch", "LocalSearch_new", "TOPSIS", "ORTools"]

    for i, device_num in enumerate(num_devices):
        # for j in range(i + 1):
        for j in range(len(num_requests)):
            request_num = num_requests[j]
            device_list = generate_devices(device_num)
            task_list = generate_workflows(request_num, device_list)
            transmission_matrix = generate_transmission_rate_matrix(len(device_list))
            for solver in solvers:
                print(f"Running i={request_num} k={device_num}, solver={solver}")
                make_decision_from_task_new(task_list, device_list, transmission_matrix, solver, display=False,
                                            record=True, iterations=10)

    # record finishes, save into csv
    df = pd.DataFrame(data)
    df.to_csv('results/evaluation_12.csv', index=False)


if __name__ == '__main__':
    main()
    # evaluation_experiments()

