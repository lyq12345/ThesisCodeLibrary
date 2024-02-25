import copy
import os
import json
import numpy as np

cur_dir = os.path.dirname(os.path.abspath(__file__))

parameters = [
  {
    "jetson-nano": {
        "a": 1,
        "b": 1
    },
    "raspberrypi-4b": {
        "a": 1,
        "b": 1
    },
    "jetson-xavier": {
        "a": 1,
        "b": 1
    },
  },
    {
        "jetson-nano": {
            "a": 1,
            "b": 1
        },
        "raspberrypi-4b": {
            "a": 1,
            "b": 1
        },
        "jetson-xavier": {
            "a": 1,
            "b": 1
        },
    },
    {
        "jetson-nano": {
            "a": 1,
            "b": 1
        },
        "raspberrypi-4b": {
            "a": 1,
            "b": 1
        },
        "jetson-xavier": {
            "a": -717208.6734812426,
            "b": -9.42588764691802e-05,
            "c": 717247.4006012462
        },
    },
    {
        "jetson-nano": {
            "a": 1,
            "b": 1
        },
        "raspberrypi-4b": {
            "a": 1,
            "b": 1
        },
        "jetson-xavier": {
            "a": 1,
            "b": 1
        },
    },
    {
        "jetson-nano": {
            "a": 1,
            "b": 1
        },
        "raspberrypi-4b": {
            "a": 1,
            "b": 1
        },
        "jetson-xavier": {
            "a": 1,
            "b": 1
        },
    },
    {
        "jetson-nano": {
            "a": 1,
            "b": 1
        },
        "raspberrypi-4b": {
            "a": 1,
            "b": 1
        },
        "jetson-xavier": {
            "a": 1,
            "b": 1
        },
    },
    {
        "jetson-nano": {
            "a": 1,
            "b": 1
        },
        "raspberrypi-4b": {
            "a": 1,
            "b": 1
        },
        "jetson-xavier": {
            "a": 1,
            "b": 1
        },
    },
    {
        "jetson-nano": {
            "a": 1,
            "b": 1
        },
        "raspberrypi-4b": {
            "a": 1,
            "b": 1
        },
        "jetson-xavier": {
            "a": 1,
            "b": 1
        },
    },
    {
        "jetson-nano": {
            "a": 1,
            "b": 1
        },
        "raspberrypi-4b": {
            "a": 1,
            "b": 1
        },
        "jetson-xavier": {
            "a": 1,
            "b": 1
        },
    },
    {
        "jetson-nano": {
            "a": 1,
            "b": 1
        },
        "raspberrypi-4b": {
            "a": 1,
            "b": 1
        },
        "jetson-xavier": {
            "a": 1,
            "b": 1
        },
    },
    {
        "jetson-nano": {
            "a": 1,
            "b": 1
        },
        "raspberrypi-4b": {
            "a": 1,
            "b": 1
        },
        "jetson-xavier": {
            "a": 1,
            "b": 1
        },
    },
    {
        "jetson-nano": {
            "a": 1,
            "b": 1
        },
        "raspberrypi-4b": {
            "a": 1,
            "b": 1
        },
        "jetson-xavier": {
            "a": 1,
            "b": 1
        },
    },
    {
        "jetson-nano": {
            "a": 1,
            "b": 1
        },
        "raspberrypi-4b": {
            "a": 1,
            "b": 1
        },
        "jetson-xavier": {
            "a": 1,
            "b": 1
        },
    },
    {
        "jetson-nano": {
            "a": 1,
            "b": 1
        },
        "raspberrypi-4b": {
            "a": 1,
            "b": 1
        },
        "jetson-xavier": {
            "a": 1,
            "b": 1
        },
    },
    {
        "jetson-nano": {
            "a": 1,
            "b": 1
        },
        "raspberrypi-4b": {
            "a": 1,
            "b": 1
        },
        "jetson-xavier": {
            "a": 1,
            "b": 1
        },
    },
]

def log_model(x, a, b):
    return a * np.log(x) + b
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