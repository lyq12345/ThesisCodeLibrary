import json
from typing import List, Dict
import os
import numpy as np
from greedy_deploy import Greedy_Decider
from MIP_deploy import MIP_Decider
from tests.status_tracker.device_mock import generate_devices

cur_dir = os.getcwd()
X = np.zeros((1000, 1000)) # operator - device
Y = np.zeros((1000, 1000)) # task - processing operator
def read_json(filename):
    with open(filename, 'r') as json_file:
        data = json.load(json_file)
    return data



def choose_best_operator(operator_candidates):
    max_speed_op = max(operator_candidates, key=lambda x: x["speed"])
    return max_speed_op



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

def make_decision_from_task_new(tasks):
    device_file = os.path.join(cur_dir, "../status_tracker/devices.json")
    operator_file = os.path.join(cur_dir, "../status_tracker/operators.json")

    # device_list = read_json(device_file)
    device_list = generate_devices(50)
    operator_list = read_json(operator_file)

    decision_maker = MIP_Decider(tasks, device_list, operator_list)
    decision_maker.make_decision()

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
# make_decison_from_tasks(tasks)
make_decision_from_task_new(tasks)

