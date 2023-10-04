import cvxpy as cp
import numpy as np
import json

class CVX_Decider:
    def __init__(self, tasks, devices):
        self.tasks = tasks
        self.devices = devices

    def read_json(self, filename):
        with open(filename, 'r') as json_file:
            data = json.load(json_file)
        return data
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

