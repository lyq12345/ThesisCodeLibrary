import copy
import os
import json
from pathlib import Path
import sys
import numpy as np
sys.path.append(str(Path(__file__).resolve().parents[1]))
from status_tracker.rescons_models import cpu_consumption
from tests.decision_maker.new.Greedy_ODP import Greedy_ODP

cur_dir = os.getcwd()

speed_lookup_table = None
power_lookup_table = None
with open(os.path.join(cur_dir, "../status_tracker/speed_lookup_table.json"), 'r') as file:
    speed_lookup_table = json.load(file)

with open(os.path.join(cur_dir, "../status_tracker/power_lookup_table.json"), 'r') as file:
    power_lookup_table = json.load(file)

class ODP_TS_Decider:
    def __init__(self, workflows, microservice_data, operator_data, devices, operators, transmission_matrix):
        self.wa = 0.05
        self.wb = 0.95
        self.workflows = workflows
        self.microservice_data = microservice_data
        """
        microservices_data = {
            "microservices_graph": None,
            "ms_wf_mapping": None,
            "ms_types": None,
        }
        """
        self.ms_to_wf = []
        for wf_id, workflow in enumerate(workflows):
            microservices = workflow["workflow"]
            for _ in microservices:
                self.ms_to_wf.append(wf_id)
        self.devices = copy.deepcopy(devices)
        self.operator_data = operator_data
        self.operator_profiles = operators
        self.operator_loads = [0 for _ in range(len(operator_data))]

        self.transmission_matrix = transmission_matrix
        self.AMax = []
        self.Amin = []
        self.calculate_max_min_acc(workflows)
        self.link_penalty_matrix = [[0 for j in range(len(devices))] for i in range(len(devices))]
        self.calculate_link_penalty(devices, transmission_matrix)

    def get_peer_operators(self, service_code, dev_name, op_load):
        candidate_op_codes = []
        for op in self.operator_profiles:
            if op["object_code"] == service_code:
                if op_load <= 1/speed_lookup_table[op["id"]][dev_name]:
                    candidate_op_codes.append(op["id"])

        return candidate_op_codes

    def calculate_max_min_acc(self, workflows):
        ms_id_global = 0
        for wf_id, workflow in enumerate(workflows):
            A_max = 1.0
            A_min = 1.0
            microservices = workflow["workflow"]
            for _ in microservices:
                min_acc = float("inf")
                max_acc = float("-inf")
                service_code = self.microservice_data["ms_types"][ms_id_global]
                for op in self.operator_profiles:
                    if op["object_code"] == service_code:
                        if op["accuracy"] > max_acc:
                            max_acc = op["accuracy"]
                        if op["accuracy"] < min_acc:
                            min_acc = op["accuracy"]
                A_max *= max_acc
                A_min *= min_acc
                ms_id_global += 1
            self.AMax.append(A_max)
            self.Amin.append(A_min)

    def calculate_link_penalty(self, devices, transmission_matrix):
        D = len(devices)
        device_speedup = {
            "raspberrypi-4b": 1,
            "jetson-nano": 1.70,
            "jetson-xavier": 2.21
        }
        D_max = 0
        D_min = float('inf')

        # get D_max and D_min
        for i in range(D):
            for j in range(0, D):
                # using human detection operator as reference operator
                dev_model1 = devices[i]["model"]
                dev_model2 = devices[j]["model"]
                D_curve = transmission_matrix[i][j] + speed_lookup_table[2][dev_model1] + speed_lookup_table[2][dev_model2]
                if D_curve > D_max:
                    D_max = D_curve
                if D_curve < D_min:
                    D_min = D_curve

        # calculate penalty
        for i in range(D):
            for j in range(0, D):
                # using human detection operator as reference operator
                dev_model1 = devices[i]["model"]
                dev_model2 = devices[j]["model"]
                D_curve = transmission_matrix[i][j] + speed_lookup_table[2][dev_model1] + speed_lookup_table[2][dev_model2]
                penalty = self.wb*((D_curve - D_min) / (D_max - D_min))
                self.link_penalty_matrix[i][j] = penalty

    def operator_reusable(self, devices, mapping, rate):
        op_id = mapping[0]
        op_code = mapping[1]
        dev_id = mapping[2]
        dev_name = devices[dev_id]["model"]
        # find operator op_id's load
        operator_load = mapping[3]
        new_load = operator_load + rate
        if new_load > 1/speed_lookup_table[op_code][dev_name]:
            return False
        cpu_extra = cpu_consumption(op_code, dev_name, new_load)-cpu_consumption(op_code, dev_name, operator_load)
        if devices[dev_id]["resources"]["system"]["cpu"]<cpu_extra:
            return False

        return True
    def calculate_resource_consumption(self, solution):
        cpu_consumptions = [0] * len(self.devices)
        ram_consumptions = [0] * len(self.devices)
        cpu_sum = 0.0
        memory_sum = 0.0
        for dev in self.devices:
            cpu_sum += dev["resources"]["system"]["cpu"]
            memory_sum += dev["resources"]["system"]["memory"]
        deployed_op_ids = []
        for i, mapping in enumerate(solution):
            op_id = mapping[0]
            if op_id in deployed_op_ids:
                continue
            op_code = mapping[1]
            dev_id = mapping[2]
            dev_name = self.devices[dev_id]["model"]
            op_load = mapping[3]

            op_resource = self.operator_profiles[op_code]["requirements"]["system"]
            cpu_cons = cpu_consumption(op_code, dev_name, op_load)
            cpu_consumptions[dev_id] += cpu_cons
            ram_consumptions[dev_id] += op_resource["memory"]
            deployed_op_ids.append(op_id)

        avg_cpu_consumption = sum(cpu_consumptions) / len(cpu_consumptions)
        avg_ram_consumption = sum(ram_consumptions) / len(ram_consumptions)
        # print(cpu_consumptions)
        # print(ram_consumptions)
        cpu_percentage = sum(cpu_consumptions) / cpu_sum

        memory_percentage = sum(ram_consumptions) / memory_sum
        return cpu_percentage, memory_percentage


    def initial_solution(self):
        greedy_decider = Greedy_ODP(self.workflows, self.microservice_data, self.operator_data, self.devices, self.operator_profiles, self.transmission_matrix, self.link_penalty_matrix)
        init_solution, init_utility = greedy_decider.make_decision(display=False)
        # self.calculate_resource_consumption(init_solution)
        return init_solution

    def swap_resource(self, devices, current_solution, neighbors):
        # replace an active device with a new resource; all operators on it should be moved
        ops_on_devices = [[] for _ in range(len(devices))]
        traversed_op_ids = []
        for mapping in current_solution:
            op_id = mapping[0]
            op_code = mapping[1]
            dev_id = mapping[2]
            op_load = mapping[3]
            if op_id not in traversed_op_ids:
                ops_on_devices[dev_id].append([op_id, op_code, op_load])
                traversed_op_ids.append(op_id)
        for dev_id, ops in enumerate(ops_on_devices):
            # find other devices that can accommodate all ops on dev_id
            filtered_dev_ids = [item for item in self.filter_devices_multiop(devices, ops) if item != dev_id]
            for other_dev_id in filtered_dev_ids:
                new_neighbor = copy.deepcopy(current_solution)
                for ms_id, mapping in enumerate(new_neighbor):
                    if mapping[2] == dev_id:
                        new_neighbor[ms_id][2] = other_dev_id
                # TODO: effective remove repeatable?
                if new_neighbor not in neighbors:
                    neighbors.append(new_neighbor)

    def move_operator(self, devices, current_solution, neighbors):
        # move an operator from u to a new v
        moved_op_ids = []
        for mapping in current_solution:
            op_id = mapping[0]
            if op_id in moved_op_ids:
                continue
            op_code = mapping[1]
            dev_id = mapping[2]
            op_load = mapping[3]
            filtered_dev_ids = [item for item in self.filter_devices(devices, (op_id, op_code, op_load)) if item != dev_id]
            for other_dev_id in filtered_dev_ids:
                new_neighbor = copy.deepcopy(current_solution)
                for mapping in new_neighbor:
                    if mapping[0] == op_id:
                        mapping[2] = other_dev_id
                # TODO: effective remove repeatable?
                if new_neighbor not in neighbors:
                    neighbors.append(new_neighbor)
            moved_op_ids.append(op_id)

    def colocate_operators(self, devices, current_solution, neighbors):
        # merge two connecting operators on the same device
        # merge the same type of microservices
        # for each microservice, try to reuse other's operator
        connecting_mappings = []
        ms_id = 0

        for wf_id, workflow in enumerate(self.workflows):
            microservices = workflow["workflow"]
            for idx in range(len(microservices)):
                if idx == 0:
                    continue
                pre_mapping = current_solution[ms_id-1]
                mapping = current_solution[ms_id]
                if mapping[2] != pre_mapping[2]:
                    connecting_mappings.append([pre_mapping, mapping])
                ms_id += 1

        for mapping1, mapping2 in connecting_mappings:
            # move to the first one
            dev_id_1 = mapping1[2]
            dev_name_1 = devices[dev_id_1]["model"]
            op_id_2 = mapping2[0]
            op_code_2 = mapping2[1]
            op_2_resource = self.operator_profiles[op_code_2]["requirements"]["system"]
            op_2_resource["cpu"] = cpu_consumption(op_code_2, dev_name_1, mapping2[3])
            if self.is_system_consistent(devices[dev_id_1]["resources"]["system"], op_2_resource):
                new_neighbor = copy.deepcopy(current_solution)
                for mapping in new_neighbor:
                    if mapping[0] == op_id_2:
                        mapping[2] = dev_id_1
                if new_neighbor not in neighbors:
                    neighbors.append(new_neighbor)

            # move to the second one
            dev_id_2 = mapping2[2]
            dev_name_2 = devices[dev_id_2]["model"]
            op_id_1 = mapping1[0]
            op_code_1 = mapping1[1]
            op_1_resource = self.operator_profiles[op_code_1]["requirements"]["system"]
            op_1_resource["cpu"] = cpu_consumption(op_code_1, dev_name_2, mapping1[3])
            if self.is_system_consistent(devices[dev_id_2]["resources"]["system"], op_1_resource):
                new_neighbor = copy.deepcopy(current_solution)
                for mapping in new_neighbor:
                    if mapping[0] == op_id_1:
                        mapping[2] = dev_id_2
                if new_neighbor not in neighbors:
                    neighbors.append(new_neighbor)

    def improve_solution(self, devices, current_solution, strategy):
        """
        exploration strategies:
        1) swap resources: replace an active device u with a new one v
        2) relocate operator: relocate a single operator i from its location u to a new device v
        3) change operator: change operator i on a device u to operator j that does the same thing
        4) microservice reuse: merge microservices with same type of requests
        """
        # moving one operator to another device; change to another operator
        best_neighbor = current_solution
        best_neighbor_utility = self.calculate_utility(current_solution)

        # print("current solution: ", current_solution)

        neighbors = []
        if strategy == 1:
            # new_device_copy = copy.deepcopy(device_copy)
            self.swap_resource(devices, current_solution, neighbors)
        elif strategy == 2:
            # new_device_copy = copy.deepcopy(device_copy)
            self.move_operator(devices, current_solution, neighbors)
        elif strategy == 3:
            # new_device_copy = copy.deepcopy(device_copy)
            self.colocate_operators(devices, current_solution, neighbors)

        for neighbor in neighbors:
            neighbor_utility = self.calculate_utility(neighbor)
            if neighbor_utility > best_neighbor_utility:
                # self.calculate_resource_consumption(neighbor)
                best_neighbor_utility = neighbor_utility
                best_neighbor = neighbor

        return best_neighbor, best_neighbor_utility

    def local_search(self, devices, current_solution):
        best_solution = current_solution
        best_utility = self.calculate_utility(current_solution)
        while True:
            current_best_utility = self.calculate_utility(best_solution)
            best_neighbor, best_neighbor_utility = self.improve_solution(devices, best_solution, 1)
            if best_neighbor_utility > best_utility:
                best_solution = best_neighbor
                best_utility = best_neighbor_utility

            best_neighbor, best_neighbor_utility = self.improve_solution(devices, best_solution, 2)
            # TODO: This will lower the search
            if best_neighbor_utility > best_utility:
                best_solution = best_neighbor
                best_utility = best_neighbor_utility

            best_neighbor, best_neighbor_utility = self.improve_solution(devices, best_solution, 3)
            if best_neighbor_utility > best_utility:
                best_solution = best_neighbor
                best_utility = best_neighbor_utility

            if best_utility <= current_best_utility:  # if no improvements made:
                break

        return best_solution, best_utility

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

    def solution_feasible(self, solution):
        device_copy = copy.deepcopy(self.devices)
        for mapping in solution:
            op_id = mapping[1]
            dev_id = mapping[2]
            resources = device_copy[dev_id]["resources"]["system"]
            requirements = self.operator_profiles[op_id]["requirements"]["system"]
            if not self.is_system_consistent(resources, requirements):
                return False
            else:
                self.deploy(device_copy, mapping)
        return True


    def filter_devices(self, devices, operator):
        op_code = operator[1]
        op_load = operator[2]
        filtered_devices = []
        for dev in devices:
            dev_name = dev["model"]
            resource_requirements = self.operator_profiles[op_code]["requirements"]["system"]
            resource_requirements["cpu"] = cpu_consumption(op_code, dev_name, op_load)
            if self.is_system_consistent(dev["resources"]["system"], resource_requirements):
                filtered_devices.append(dev)
        filtered_device_ids = [d["id"] for d in filtered_devices]
        return filtered_device_ids

    def filter_devices_multiop(self, devices, operators):
        filtered_devices = []
        for dev in devices:
            dev_name = dev["model"]
            accumulated_resources = {
                "cpu": 0,
                "gpu": 0,
                "storage": 0,
                "memory": 0
            }
            for op_id, op_code, op_load in operators:
                operator = self.operator_profiles[op_code]
                for key, value in operator["requirements"]["system"].items():
                    if key != "cpu":
                        accumulated_resources[key] += value
                    else:
                        accumulated_resources[key] += cpu_consumption(op_code, dev_name, op_load)
            if self.is_system_consistent(dev["resources"]["system"], accumulated_resources):
                filtered_devices.append(dev)
        filtered_device_ids = [d["id"] for d in filtered_devices]
        return filtered_device_ids

    def calculate_utility(self, solution):
        sum_uti = 0
        ms_id = 0
        for wf_id, workflow in enumerate(self.workflows):
            source_device_id = workflow["source"]
            delay_tol = workflow["delay"]
            acc_max = self.AMax[wf_id]
            acc_min = self.Amin[wf_id]
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
                if i == 0:  # the first microservice
                    delay += self.transmission_matrix[source_device_id][dev_id]
                else:
                    previous_dev_id = solution[ms_id - 1][2]
                    delay += self.transmission_matrix[previous_dev_id][dev_id]
                ms_id += 1
            # wa = 0.05
            # wb = 0.95
            if acc_max == acc_min:
                A = accuracy
            else:
                A = (accuracy-acc_min)/(acc_max-acc_min)
            B = (delay_tol - delay) / delay_tol
            utility = self.wa*A + self.wb*B
            sum_uti += utility
        return sum_uti

    def calculate_power(self, operator, device_id):
        operator_name = operator["name"]
        device_model = self.devices[device_id]["model"]
        power = power_lookup_table[operator_name][device_model]
        return power

    def deploy(self, devices, mapping):
        op_code = mapping[1]
        dev_id = mapping[2]
        dev_name = self.devices[dev_id]["model"]
        op_load = mapping[3]
        resource_requirements = self.operator_profiles[op_code]["requirements"]["system"]
        resource_requirements["cpu"] = cpu_consumption(op_code, dev_name, op_load)

        for type, amount in resource_requirements.items():
            devices[dev_id]["resources"]["system"][type] -= amount

    def undeploy(self, devices, mapping):
        op_code = mapping[1]
        dev_id = mapping[2]
        dev_name = self.devices[dev_id]["model"]
        op_load = mapping[3]
        resource_requirements = self.operator_profiles[op_code]["requirements"]["system"]
        resource_requirements["cpu"] = cpu_consumption(op_code, dev_name, op_load)

        for type, amount in resource_requirements.items():
            devices[dev_id]["resources"]["system"][type] += amount

    def save_dict_to_json(self, dictionary, filename):
        with open(filename, 'w') as json_file:
            json.dump(dictionary, json_file, indent=4)

    # 保存列表为 JSON 文件
    def save_list_to_json(self, lst, filename):
        with open(filename, 'w') as json_file:
            json.dump(lst, json_file, indent=4)

    def consume_operators(self, devices, solution):
        deployed_op_ids = []
        for mapping in solution:
            if mapping[0] in deployed_op_ids:
                continue
            self.deploy(devices, mapping)
            deployed_op_ids.append(mapping[0])

    def make_decision(self):
        print("Running ODP-TS decision maker")
        initial_solution = self.initial_solution()
        device_copy = copy.deepcopy(self.devices)
        self.consume_operators(device_copy, initial_solution)
        S1, F1 = self.local_search(device_copy, initial_solution) # local optima
        F_star = float("-inf")
        S_star = None
        S = S1
        tabu_limit = 100
        tabu_list = []

        counter = 0
        max_non_improvement = 10
        while counter <= max_non_improvement:
            improvement = False
            device_copy = copy.deepcopy(self.devices)
            self.consume_operators(device_copy, S)
            S, F = self.local_search(device_copy, S)  # best neighbor
            # F = self.calculate_utility(S)
            if F == F_star and S not in tabu_list:
                tabu_list.append(S)
                if len(tabu_list) > tabu_limit:
                    tabu_list.pop(0)
            if F > F_star and S not in tabu_list:
                S_star = S
                F_star = F
                tabu_list.append(S)
                if len(tabu_list) > tabu_limit:
                    tabu_list.pop(0)
                improvement = True
            if not improvement:
                counter += 1
        if F1 > F_star:
            S_star = S1
        return S_star, F_star


