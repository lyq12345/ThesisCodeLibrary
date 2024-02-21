import copy
import os
import json

cur_dir = os.path.dirname(os.path.abspath(__file__))
def read_json(filename):
    with open(filename, 'r') as json_file:
        data = json.load(json_file)
    return data
def cpu_consumption(op_code, dev_model, load):
    operator_file = os.path.join(cur_dir, "operators.json")
    # operator_file = "operators.json"
    operator_list = read_json(operator_file)
    if load<=0.02:
        return operator_list[op_code]["requirements"]["system"]["cpu"]
    else:
        return operator_list[op_code]["requirements"]["system"]["cpu"] + (load/0.02)*20