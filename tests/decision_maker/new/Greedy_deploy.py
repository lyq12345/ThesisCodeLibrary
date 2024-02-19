import numpy as np
import math
import copy
import os
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from status_tracker.rescons_models import cpu_consumption

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
    def __init__(self, workflows, microservice_data, operator_data, devices, operators, transmission_matrix, policy="accfirst"):
        self.workflows = workflows
        self.policy = policy
        self.microservice_data = microservice_data
        """
        microservices_data = {
            "microservices_graph": None,
            "ms_wf_mapping": None,
            "ms_types": None,
        }
        """
        self.devices = copy.deepcopy(devices)
        self.original_devices = copy.deepcopy(devices)
        self.operator_data = operator_data
        self.operator_profiles = operators
        self.operator_loads = [0.0 for _ in range(len(operator_data))]

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

    def filter_devices(self, op_code):
        filtered_devices = []
        operator = self.operator_profiles[op_code]
        for dev in self.devices:
            if self.is_system_consistent(dev["resources"]["system"], operator["requirements"]["system"]):
                filtered_devices.append(dev)
        filtered_device_ids = [d["id"] for d in filtered_devices]
        return filtered_device_ids

    def calculate_utility(self, solution):
        sum_uti = 0
        ms_id = 0
        for wf_id, workflow in enumerate(self.workflows):
            source_device_id = workflow["source"]
            delay_tol = workflow["delay"]
            accuracy = 1
            delay = 0

            for i in range(len(workflow["workflow"])):
                mapping = solution[ms_id]
                op_code = mapping[1]
                dev_id = mapping[2]
                dev_name = self.devices[dev_id]["model"]
                op_accuracy = self.operator_profiles[op_code]["accuracy"]
                accuracy *= op_accuracy
                operator_delay = speed_lookup_table[op_code][dev_name]
                delay += operator_delay
                if i == 0: # the first microservice
                    delay += self.transmission_matrix[source_device_id][dev_id]
                else:
                    previous_dev_id = solution[ms_id-1][2]
                    delay += self.transmission_matrix[previous_dev_id][dev_id]
                ms_id += 1
            utility = 0.12*accuracy - 0.88*max(0, (delay - delay_tol)/delay)
            sum_uti += utility
        return sum_uti

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

    def calculate_resource_consumption(self, solution):
        cpu_consumptions = [0] * len(self.original_devices)
        ram_consumptions = [0] * len(self.original_devices)
        deployed_op_ids = []
        for i, mapping in enumerate(solution):
            op_id = mapping[0]
            if op_id in deployed_op_ids:
                continue
            op_code = mapping[1]
            dev_id = mapping[2]
            dev_name = self.original_devices[dev_id]["model"]
            op_load = mapping[3]

            op_resource = self.operator_profiles[op_code]["requirements"]["system"]
            cpu_cons = cpu_consumption(op_code, dev_name, op_load)
            cpu_consumptions[dev_id] += cpu_cons
            ram_consumptions[dev_id] += op_resource["memory"]
            deployed_op_ids.append(op_id)
        for i in range(len(cpu_consumptions)):
            cpu_consumptions[i] = cpu_consumptions[i] / self.original_devices[i]["resources"]["system"]["cpu"]
            ram_consumptions[i] = ram_consumptions[i] / self.original_devices[i]["resources"]["system"]["memory"]
        print("Inside Greedy Solver: ")
        print("CPU consumptions: ")
        print(cpu_consumptions)
        print("Memory consumptions:")
        print(ram_consumptions)
        avg_cpu_consumption = sum(cpu_consumptions) / len(cpu_consumptions)
        avg_ram_consumption = sum(ram_consumptions) / len(ram_consumptions)
        return avg_cpu_consumption, avg_ram_consumption

    def deploy(self, devices, mapping, rate):
        # resource consumption
        op_id = mapping[0]
        op_code = mapping[1]
        dev_id = mapping[2]
        dev_name = devices[dev_id]["model"]
        operator_resource = self.operator_profiles[op_code]["requirements"]["system"]

        for type, amount in operator_resource.items():
            if type != "cpu":
                devices[dev_id]["resources"]["system"][type] -= amount
            else:
                devices[dev_id]["resources"]["system"][type] -= cpu_consumption(op_code, dev_name, rate)
        self.operator_loads[op_id] += rate

    def reuse(self, devices, mapping, rate):
        op_id = mapping[0]
        op_code = mapping[1]
        dev_id = mapping[2]
        dev_name = devices[dev_id]["model"]
        previous_load = self.operator_loads[op_id]
        new_load = previous_load + rate
        extra_cpu = cpu_consumption(op_code, dev_name, new_load) - cpu_consumption(op_code, dev_name, previous_load)

        devices[dev_id]["resources"]["system"]["cpu"] -= extra_cpu
        self.operator_loads[op_id] += rate
    def find_best_mapping_acc(self, mapping_candidates, previous_mapping):
        # accuracy first
        best_accuracy = 0
        best_mapping = None
        high_acc_candidates = []
        for id, mapping in enumerate(mapping_candidates):
            op_code = mapping[1]
            accuracy = self.operator_profiles[op_code]["accuracy"]
            if accuracy > best_accuracy:
                best_accuracy = accuracy
        for mapping in mapping_candidates:
            op_code = mapping[1]
            accuracy = self.operator_profiles[op_code]["accuracy"]
            if accuracy == best_accuracy:
                high_acc_candidates.append(mapping)

        lowest_delay = float("inf")
        for mapping in high_acc_candidates:
            op_code = mapping[1]
            dev_id = mapping[2]
            dev_name = self.devices[dev_id]["model"]
            delay = 0.0
            delay += speed_lookup_table[op_code][dev_name] # operator latency
            if previous_mapping[0] == -1:
                source_dev_id = previous_mapping[2]
                delay += self.transmission_matrix[source_dev_id][dev_id]
            else:
                previous_dev_id = previous_mapping[2]
                delay += self.transmission_matrix[previous_dev_id][dev_id]
            if delay < lowest_delay:
                best_mapping = mapping
                lowest_delay = delay

        return best_mapping

    def find_best_mapping_delay(self, mapping_candidates, previous_mapping):
        # accuracy first
        lowest_delay = float("inf")
        best_mapping = None
        low_acc_candidates = []
        for id, mapping in enumerate(mapping_candidates):
            op_code = mapping[1]
            dev_id = mapping[2]
            dev_name = self.devices[dev_id]["model"]
            delay = 0.0
            delay += speed_lookup_table[op_code][dev_name]  # operator latency
            if previous_mapping[0] == -1:
                source_dev_id = previous_mapping[2]
                delay += self.transmission_matrix[source_dev_id][dev_id]
            else:
                previous_dev_id = previous_mapping[2]
                delay += self.transmission_matrix[previous_dev_id][dev_id]
            if delay < lowest_delay:
                lowest_delay = delay
        for mapping in mapping_candidates:
            op_code = mapping[1]
            dev_id = mapping[2]
            dev_name = self.devices[dev_id]["model"]
            delay = 0.0
            delay += speed_lookup_table[op_code][dev_name]  # operator latency
            if previous_mapping[0] == -1:
                source_dev_id = previous_mapping[2]
                delay += self.transmission_matrix[source_dev_id][dev_id]
            else:
                previous_dev_id = previous_mapping[2]
                delay += self.transmission_matrix[previous_dev_id][dev_id]
            if delay == lowest_delay:
                low_acc_candidates.append(mapping)

        highest_accuracy = 0
        for mapping in low_acc_candidates:
            op_code = mapping[1]
            accuracy = self.operator_profiles[op_code]["accuracy"]
            if accuracy > highest_accuracy:
                best_mapping = mapping
                highest_accuracy = accuracy

        return best_mapping
    def find_best_mapping_multi(self, mapping_candidates, previous_mapping):
        best_mapping = None
        decision_matrix = []
        num_criterias = 2
        for mapping in mapping_candidates:
            op_code = mapping[1]
            dev_id = mapping[2]
            dev_name = self.devices[dev_id]["model"]
            accuracy = self.operator_profiles[op_code]["accuracy"]
            delay = 0.0
            delay += speed_lookup_table[op_code][dev_name]  # operator latency
            if previous_mapping[0] == -1:
                source_dev_id = previous_mapping[2]
                delay += self.transmission_matrix[source_dev_id][dev_id]
            else:
                previous_dev_id = previous_mapping[2]
                delay += self.transmission_matrix[previous_dev_id][dev_id]
            criteria_list = [accuracy, delay]
            decision_matrix.append(criteria_list)

        decision_matrix_np = np.array(decision_matrix)

        # Calculate Weighted Normalized Decision Matrix (WNDM)
        weights = [0.5, 0.5]
        # Calculate Normalized Decision Matrix (NDM)
        for i in range(len(decision_matrix)):
            for j in range(num_criterias):
                decision_matrix_np[i, j] = decision_matrix_np[i, j] * weights[j]

        # Determine the best solution (A+) and the worst solution (A−)
        max_of_accuracy = np.max(decision_matrix_np[:, 0])
        min_of_accuracy = np.min(decision_matrix_np[:, 0])
        min_of_delay = np.min(decision_matrix_np[:, 1])
        max_of_delay = np.max(decision_matrix_np[:, 1])

        A_plus = np.array([max_of_accuracy, min_of_delay])
        A_minus = np.array([min_of_accuracy, max_of_delay])

        # Calculate the Separation Measures (SM).
        SM_plus = np.zeros(len(decision_matrix))
        SM_minus = np.zeros(len(decision_matrix))
        for i in range(len(decision_matrix)):
            sum_square_plus = 0
            sum_square_minus = 0
            for j in range(num_criterias):
                sum_square_plus += (decision_matrix_np[i, j] - A_plus[j]) ** 2
                sum_square_minus += ((decision_matrix_np[i, j] - A_minus[j]) ** 2)
            SM_plus[i] = math.sqrt(sum_square_plus)
            SM_minus[i] = math.sqrt(sum_square_minus)

        RC = np.zeros(len(decision_matrix))
        for i in range(len(decision_matrix)):
            RC[i] = SM_minus[i] / (SM_plus[i] + SM_minus[i])

        max_rc = max(RC)
        selected_mapping_id = -1
        for i in range(len(decision_matrix)):
            if RC[i] == max_rc:
                selected_mapping_id = i
                break
        if selected_mapping_id != -1:
            best_mapping = mapping_candidates[selected_mapping_id]
        return best_mapping

    def operator_reusable(self, mapping, rate):
        op_id = mapping[0]
        op_code = mapping[1]
        dev_id = mapping[2]
        dev_name = self.devices[dev_id]["model"]
        # find operator op_id's load
        operator_load = self.operator_loads[op_id]
        new_load = operator_load + rate
        if new_load > 1/speed_lookup_table[op_code][dev_name]:
            return False
        cpu_extra = cpu_consumption(op_code, dev_name, new_load) - cpu_consumption(op_code, dev_name, operator_load)
        if self.devices[dev_id]["resources"]["system"]["cpu"] < cpu_extra:
            return False

        return True

    def make_decision(self, display=True):
        if display:
            print("Running Greedy decision maker")
        """
        solution format:
        [[op_id, op_type, dev_id], [op_id, op_type, dev_id], [op_id, op_type, dev_id], ...]
        """
        solution = [[] for _ in range(len(self.microservice_data["ms_types"]))]
        for wf_id, workflow in enumerate(self.workflows):
            source_device_id = workflow["source"]
            delay_tol = workflow["delay"]
            rate = workflow["rate"]
            ms_ids = []
            for id in range(len(self.microservice_data["ms_types"])):
                if(self.microservice_data["ms_wf_mapping"][id][wf_id] == 1):
                    ms_ids.append(id)
            previous_mapping = [-1, -1, source_device_id, 0.0]
            for ms_id in ms_ids:
                service_code = self.microservice_data["ms_types"][ms_id]
                mapping_candidates = []
                undeployed_ops = []
                # traverse all operators
                for op_id, op_code in enumerate(self.operator_data):
                    if self.operator_profiles[op_code]["object_code"] == service_code:
                        # check if this operator is deployed
                        if self.operator_loads[op_id] > 0:
                            for mapping in solution:
                                if len(mapping)>0 and mapping[0] == op_id and self.operator_reusable(mapping, rate):
                                    mapping_candidates.append(mapping)
                                    break
                        else:
                            undeployed_ops.append([op_id, self.operator_data[op_id]])
                seen = set()
                undeployed_ops_nodup = []
                for op_id, op_code in undeployed_ops:
                    if op_code in seen:
                        continue
                    seen.add(op_code)
                    undeployed_ops_nodup.append([op_id, op_code])
                # Add candidate mappings
                for op_id, op_code in undeployed_ops_nodup:
                    filtered_dev_ids = self.filter_devices(op_code)
                    for dev_id in filtered_dev_ids:
                        mapping_candidates.append([op_id, op_code, dev_id, 0.0])
                if len(mapping_candidates) == 0:
                    print("No mapping found!")
                best_mapping = None
                if self.policy == "accfirst":
                    best_mapping = self.find_best_mapping_acc(mapping_candidates, previous_mapping)
                elif self.policy == "delayfirst":
                    best_mapping = self.find_best_mapping_delay(mapping_candidates, previous_mapping)
                elif self.policy == "multi":
                    best_mapping = self.find_best_mapping_multi(mapping_candidates, previous_mapping)

                if best_mapping is None:
                    print("No best mapping found!")
                if best_mapping[0] in [item[0] for item in solution if len(item)>0]:
                    self.reuse(self.devices, best_mapping, rate)
                else:
                    self.deploy(self.devices, best_mapping, rate)
                best_mapping[3] = self.operator_loads[best_mapping[0]]
                solution[ms_id] = best_mapping
                previous_mapping = best_mapping

        # reuse_count = {}
        # reused_ops = []
        # for mapping in solution:
        #     op_id = mapping[0]
        #     if op_id in reuse_count:
        #         if reuse_count[op_id] == 1:
        #             reused_ops.append(op_id)
        #     reuse_count[op_id] = reuse_count.get(op_id, 0) + 1

        for mapping in solution:
            mapping[3] = self.operator_loads[mapping[0]]

        # self.calculate_resource_consumption(solution)

        utility = self.calculate_utility(solution)
        return solution, utility




