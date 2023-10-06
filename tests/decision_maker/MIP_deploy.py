import cvxpy as cp
import numpy as np
import json
import os

speed_lookup_table = {
  "joelee0515/firedetection:yolov3-measure-time": {
    "jetson-nano": 4.364,
    "raspberrypi-4b": 7.0823,
    "jetson-xavier": 2.6235

  },
  "joelee0515/firedetection:tinyyolov3-measure-time": {
    "jetson-nano": 0.5549,
    "raspberrypi-4b": 1.0702,
    "jetson-xavier": 0.4276
  },
  "joelee0515/humandetection:yolov3-measure-time": {
    "jetson-nano": 4.4829,
    "raspberrypi-4b": 7.2191,
    "jetson-xavier": 3.8648
  },
  "joelee0515/humandetection:tinyyolov3-measure-time": {
    "jetson-nano": 0.5864,
    "raspberrypi-4b": 1.0913,
    "jetson-xavier": 0.4605
  }
}

class MIP_Decider:
    def __init__(self, tasks, devices, operators):
        self.tasks = tasks
        self.device_data = self.create_device_model(devices)
        self.operator_data = self.create_operator_model(tasks, devices, operators)

    def create_device_model(self, devices):
        data = {}
        """Stores the data for the problem."""
        data["resource_capability"] = []

        for device in devices:
            data["resource_capability"].append([device["resources"]["system"][key] for key in device["resources"]["system"]])

        data["transmission_speed"] = [[0, 6.43163439, 19.35637777, 15.13368045, 13.53071896, 11.69206187],
                                      [6.43163439, 0, 17.22729904, 18.4682481, 5.52004585, 8.88291174],
                                      [19.35637777, 17.22729904, 0, 16.86937148, 10.66980807, 13.09753162],
                                      [15.13368045, 18.4682481, 10.66980807, 0, 14.25380426, 14.14231382],
                                      [13.53071896, 5.52004585, 13.09753162, 14.25380426, 0, 6.70848206],
                                      [11.69206187, 8.88291174, 9.19870435, 14.14231382, 6.70848206, 0]]

        print(data)

        return data

    def create_operator_model(self, tasks, devices, operators):
        data = {}
        num_devices = len(devices)
        num_tasks = len(tasks)
        data["operator_accuracies"] = []
        data["resource_requirements"] = []
        data["processing_speed"] = []
        data["power_consumptions"] = []
        for task in tasks:
            object_type = task["object"]
            data_source = task["source"]
            device_names = [dev["model"] for dev in devices]
            for op in operators:
                if op["type"] == "processing" and op["object"] == object_type:
                    op_name = op["name"]
                    data["operator_accuracies"].append(op["accuracy"])
                    data["resource_requirements"].append([op["requirements"]["system"][key] for key in op["requirements"]["system"]])

                    data["processing_speed"].append([speed_lookup_table[op_name][dev_name] for dev_name in device_names])
                    data["power_consumptions"].append([speed_lookup_table[op_name][dev_name] for dev_name in device_names])

        print(data)
        return data


    def inverse(self, x):
        if x == 0:
            return 0
        else:
            return 1 / x

    def is_hardware_consistent(self, hardware_resources, hardware_requirements):
        if hardware_requirements is None:
            return True
        if hardware_requirements is not None and hardware_resources is None:
            return False
        for requirement in hardware_requirements:
            flag = False
            for resource in hardware_resources:
                if resource["sensor"] == requirement["sensor"]:
                    flag = True
            if not flag:
                return False
        return True
    def is_system_consistent(self, system_resources, system_requirements):
        for key, value in system_requirements.items():
            if key not in system_resources:
                return False
            if key in system_resources:
                if isinstance(value, int) or isinstance(value, float):
                    if system_resources[key] < system_resources[key]:
                        return False
                else:
                    if system_requirements[key] != system_resources[key]:
                        return False

        return True

