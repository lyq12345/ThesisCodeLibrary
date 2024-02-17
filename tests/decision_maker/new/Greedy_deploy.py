import numpy as np
import math
import copy
import os
import json
import heapq

cur_dir = os.getcwd()

speed_lookup_table = None
power_lookup_table = None
with open(os.path.join(cur_dir, "../status_tracker/speed_lookup_table.json"), 'r') as file:
    speed_lookup_table = json.load(file)

with open(os.path.join(cur_dir, "../status_tracker/power_lookup_table.json"), 'r') as file:
    power_lookup_table = json.load(file)

"""
For each workflow:
    greedily find the device with the lowest transmission delay with the last node
    for each microservice mi:
        keep satisfying operators on di based on:
            1. Accuracy first;
            2. Speed first;
            3. TOPSIS(multi-cretiria)
        until it's full; 
"""
class Greedy_decider:
    def __init__(self, workflows, microservice_data, operator_data, devices, operators, transmission_matrix):
        self.workflows = workflows
        self.microservice_data = microservice_data
        """
        microservices_data = {
            "microservices_graph": None,
            "ms_wf_mapping": None,
            "ms_types": None,
        }
        """
        self.devices = copy.deepcopy(devices)
        self.operator_data = operator_data
        self.operator_profiles = operators
        self.operator_loads = [0*len(operator_data)]

        self.transmission_matrix = transmission_matrix

    def is_system_consistent(self, system_resources, system_requirements):
        for key, value in system_requirements.items():
            if key not in system_resources:
                return False
            if key in system_resources:
                if isinstance(value, int) or isinstance(value, float):
                    if system_resources[key] < system_requirements[key]:
                        return False
                else:
                    if system_requirements[key] != system_resources[key]:
                        return False

        return True

    def filter_devices(self, operator_id):
        filtered_devices = []
        operator = self.operator_profiles[operator_id]
        for dev in self.devices:
            if self.is_system_consistent(dev["resources"]["system"], operator["requirements"]["system"]):
                filtered_devices.append(dev)
        filtered_device_ids = [d["id"] for d in filtered_devices]
        return filtered_device_ids

    def calculate_utility(self, solution):
        sum_uti = 0
        for wf_id, mapping in enumerate(solution):
            source_device_id = self.workflows[wf_id]["source"]
            operator_id = mapping[1]
            device_id = mapping[2]
            accuracy = self.operator_profiles[operator_id]["accuracy"]
            delay = self.calculate_delay(operator_id, source_device_id, device_id)
            task_del = self.workflows[wf_id]["delay"]
            utility = accuracy - max(0, (delay - task_del) / delay)
            sum_uti += utility
        cost = sum_uti
        return cost

    def calculate_delay(self, operator_id, source_device_id, device_id):
        device_model = self.devices[device_id]["model"]
        transmission_delay = self.transmission_matrix[source_device_id, device_id]
        processing_delay = speed_lookup_table[operator_id][device_model]
        return transmission_delay + processing_delay

    def calculate_power(self, operator_id, device_id):
        # operator_name = self.operators[operator_id]["name"]
        device_model = self.devices[device_id]["model"]
        power = power_lookup_table[operator_id][device_model]
        return power

    def deploy(self, devices, mapping):
        # resource consumption
        operator_code = mapping[1]
        device_id = mapping[2]
        operator_resource = {}
        for op in self.operator_profiles:
            if operator_code == op["id"]:
                operator_resource = op["requirements"]["system"]

        for type, amount in operator_resource.items():
            devices[device_id]["resources"]["system"][type] -= amount

    def reuse(self, op_id, ms_id):
        pass

    # nearest device other than itself
    def nearest_device(self, transmission_matrix, current):
        nearest_distance = float("inf")
        nearest_device = current
        for dev_id in range(len(self.devices)):
            if dev_id != current and transmission_matrix[current][dev_id] < nearest_distance:
                nearest_distance = transmission_matrix[current][dev_id]
                nearest_device = dev_id
        return nearest_device

    def operator_reusable(self, mapping, rate):
        op_id = mapping[0]
        op_code = mapping[1]
        dev_id = mapping[2]
        dev_name = self.devices[dev_id]["model"]
        # find operator op_id's load
        operator_load = self.operator_loads[op_id]
        new_load = operator_load + rate
        if new_load > speed_lookup_table[op_code][dev_id][dev_name]:
            return False
        cpu_extra = 0
        if self.devices[dev_id]["resources"]["system"]["cpu"]<cpu_extra:
            return False

        return True

    def make_decision(self, display=True):
        if display:
            print("Running Greedy decision maker")
        """
        solution format:
        [[op_id, op_type, dev_id], [op_id, op_type, dev_id], [op_id, op_type, dev_id], ...]
        """
        solution = [[] * len(self.microservice_data["ms_types"])]
        for wf_id, workflow in enumerate(self.workflows):
            source_device_id = workflow["source"]
            delay_tol = workflow["delay"]
            ms_ids = []
            for id in range(len(self.microservice_data["ms_types"])):
                if(self.microservice_data["ms_wf_mapping"][id][wf_id] == 1):
                    ms_ids.append(id)
            for ms_id in ms_ids:
                object_code = self.microservice_data["ms_types"][ms_id]
                for mapping in solution:
                    if len(mapping) > 0 and self.operator_profiles[mapping[1]]["object_code"]==object_code:
                        if self.operator_reusable(mapping[0], solution):
                            pass

        utility = self.calculate_utility(solution)
        return solution, utility




