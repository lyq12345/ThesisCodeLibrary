import json
import time
import os
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from greedy_deploy import Greedy_Decider
from MIP_deploy import MIP_Decider
from TOPSIS_deploy import TOPSIS_decider
from LocalSearch_deploy import LocalSearch_deploy

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


def generate_transmission_rate_matrix(n, min_rate=5, max_rate=15):
    transmission_matrix = np.full((n, n), np.inf)

    # 对角线上的元素设为0
    np.fill_diagonal(transmission_matrix, 0)

    # 随机生成不同device之间的传输速率并保持对称性
    for i in range(n):
        for j in range(i + 1, n):
            rate = np.random.randint(min_rate, max_rate + 1)  # 生成随机速率
            transmission_matrix[i, j] = rate
            transmission_matrix[j, i] = rate  # 对称性

    return transmission_matrix

def make_decison_from_tasks(task_list):
    device_file = os.path.join(cur_dir, "../status_tracker/devices.json")
    operator_file = os.path.join(cur_dir, "../status_tracker/operators.json")
    source_file = os.path.join(cur_dir, "../status_tracker/sources.json")

    device_list = read_json(device_file)
    operator_list = read_json(operator_file)
    source_dict = read_json(source_file)

    operator_pairs = []

    decision_maker = Greedy_Decider()


    for task in sorted(task_list, key=lambda x: x["priority"], reverse=True):
        # query for operator
        selected_sop = {}
        selected_pop = {}

        pop_candidates = []

        source_id = task["source"]
        model = source_dict[source_id]
        object = task['object']

        # query for operator
        for op in operator_list:
            if op["type"] == "source" and op["sensor"] == model:
                selected_sop = op

            if op["type"] == "processing" and op["object"] == object:
                pop_candidates.append(op)

        # greedly look for the most accurate operator that is within the delay tolerance
        selected_pop = choose_best_operator(pop_candidates)

        operator_pairs.append({"source": selected_sop, "processing": selected_pop})

    solution = decision_maker.match_operators_with_devices(operator_pairs, device_list)

def make_decision_from_task_new(task_list, device_list, transmission_matrix, solver="LocalSearch"):
    operator_file = os.path.join(cur_dir, "../status_tracker/operators.json")
    operator_list = read_json(operator_file)

    def calculate_accuracy(operator_id):
        return  operator_list[operator_id]["accuracy"]

    def calculate_delay(operator_id, source_device_id, device_id):
        device_model = device_list[device_id]["model"]
        transmission_delay = transmission_matrix[source_device_id, device_id]
        processing_delay = speed_lookup_table[operator_id][device_model]
        return transmission_delay + processing_delay

    def calculate_power(operator_id, device_id):
        operator_name = operator_list[operator_id]["name"]
        device_model = device_list[device_id]["model"]
        power = power_lookup_table[operator_name][device_model]
        return power


    # decision_maker = MIP_Decider(task_list, device_list, operator_list)
    decision_maker = None
    if solver == "TOPSIS":
        decision_maker = TOPSIS_decider(task_list, device_list, operator_list, transmission_matrix)
    elif solver == "LocalSearch":
        decision_maker = LocalSearch_deploy(task_list, device_list, operator_list, transmission_matrix)
    elif solver == "MIP":
        decision_maker = MIP_Decider(task_list, device_list, operator_list, transmission_matrix)
    start_time = time.time()
    solution, utility = decision_maker.make_decision()
    end_time = time.time()
    elapsed_time = end_time - start_time
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
        print(f"Data flow {i}:")
        print(f"Task: object: {task_object}, delay tolerance: {task_delay}")
        print(f"Deployment: {op_name} -> device {dev_id}")
        print(f"Performance: accuracy: {performance_acc}, delay: {performance_delay}, power: {performance_power}")
        print("--------------------------------------------------------------")
    print(f"Decision making time: {elapsed_time} s")
    print(f"Objective: {utility}")

tasks = [
    {
        "id": 0,
        "source": 1,
        "object": "human",
        "object_code": 1,
        "delay": 10,
        "priority": 10
    },
    {
        "id": 1,
        "source": 2,
        "object": "fire",
        "object_code": 2,
        "delay": 10,
        "priority": 5
    },
    {
        "id": 2,
        "source": 3,
        "object": "fire",
        "object_code": 2,
        "delay": 10,
        "priority": 2
    },
    {
        "id": 3,
        "source": 4,
        "object_code": 2,
        "object": "fire",
        "delay": 10,
        "priority": 1
    }
]

num_devices = 40
num_tasks = 4

if len(sys.argv) != 3:
    print("not enough parameters")
else:
    # 获取命令行参数
    arg1 = sys.argv[1]
    arg2 = sys.argv[2]

    num_devices = int(arg1)
    num_tasks = int(arg2)

# make_decison_from_tasks(tasks)
device_list = generate_devices(num_devices)
# for dev in device_list:
#     print(dev["model"])
task_list = generate_tasks(num_tasks, device_list)
transmission_matrix = generate_transmission_rate_matrix(len(device_list))

make_decision_from_task_new(task_list, device_list, transmission_matrix, "LocalSearch")
make_decision_from_task_new(task_list, device_list, transmission_matrix, "TOPSIS")
make_decision_from_task_new(task_list, device_list, transmission_matrix, "MIP")


