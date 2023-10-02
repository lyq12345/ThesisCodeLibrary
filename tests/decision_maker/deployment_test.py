import json
from typing import List, Dict
import os
import numpy as np
from greedy_deploy import Greedy_Decider

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

tasks = [
    {
        "id": 0,
        "source": "1",
        "object": "human",
        "delay": 10,
        "priority": 10
    },
    {
        "id": 1,
        "source": "2",
        "object": "fire",
        "delay": 10,
        "priority": 5
    },
    {
        "id": 2,
        "source": "2",
        "object": "fire",
        "delay": 10,
        "priority": 2
    },
    {
        "id": 3,
        "source": "2",
        "object": "fire",
        "delay": 10,
        "priority": 1
    },
]
make_decison_from_tasks(tasks)

