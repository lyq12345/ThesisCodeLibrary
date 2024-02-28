import copy
import os
import json
from pathlib import Path
import sys
import numpy as np
sys.path.append(str(Path(__file__).resolve().parents[1]))
from status_tracker.rescons_models import cpu_consumption
from tests.decision_maker.new.Greedy_deploy import Greedy_decider
import math
import random

cur_dir = os.getcwd()

speed_lookup_table = None
power_lookup_table = None
with open(os.path.join(cur_dir, "../status_tracker/speed_lookup_table.json"), 'r') as file:
    speed_lookup_table = json.load(file)

with open(os.path.join(cur_dir, "../status_tracker/power_lookup_table.json"), 'r') as file:
    power_lookup_table = json.load(file)

class SA_Decider:
    def __init__(self, workflows, microservice_data, operator_data, devices, operators, transmission_matrix, iter=10, T0=100, Tf=1e-8, alpha=0.99):
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

        self.iter = iter
        self.alpha = alpha
        self.T0 = T0
        self.Tf = Tf
        self.T = T0

        self.history = {"f": [], "T": []}

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

    def Metrospolis(self, f, f_new):
        if f_new <= f:
            return 1
        else:
            p = math.exp((f - f_new) / self.T)
            if random.random() < p:
                return 1
            else:
                return 0

    def get_peer_operators(self, service_code, dev_name, op_load):
        candidate_op_codes = []
        for op in self.operator_profiles:
            if op["object_code"] == service_code:
                if op_load <= 1/speed_lookup_table[op["id"]][dev_name]:
                    candidate_op_codes.append(op["id"])

        return candidate_op_codes

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
        # devices_copy = copy.deepcopy(self.devices)
        # for task in self.tasks:
        #     task_id = task["id"]
        #     candidate_op_ids = self.get_candidate_operators(task)
        #     selected_op_id = random.choice(candidate_op_ids)
        #     candidate_device_ids = self.filter_devices(devices_copy, selected_op_id)
        #     selected_device_id = random.choice(candidate_device_ids)
        #     self.deploy(devices_copy, (selected_op_id, selected_device_id))
        #     self.solution[task_id] = (selected_op_id, selected_device_id)

        greedy_decider = Greedy_decider(self.workflows, self.microservice_data, self.operator_data, self.devices, self.operator_profiles, self.transmission_matrix, "multi" )
        init_solution, init_utility = greedy_decider.make_decision(display=False)
        # self.calculate_resource_consumption(init_solution)
        return init_solution

    def swap_resource(self, devices, current_solution):
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

        counter = 0
        while counter <= 10:
            # randomly choose one device and swap with another device
            random_dev_id = random.randint(0, len(ops_on_devices)-1)
            ops = ops_on_devices[random_dev_id]
            filtered_dev_ids = [item for item in self.filter_devices_multiop(devices, ops) if item != random_dev_id]
            if len(filtered_dev_ids) == 0:
                counter += 1
                continue
            other_random_dev_id = random.choice(filtered_dev_ids)
            new_neighbor = copy.deepcopy(current_solution)
            for ms_id, mapping in enumerate(new_neighbor):
                if mapping[2] == random_dev_id:
                    new_neighbor[ms_id][2] = other_random_dev_id
            return new_neighbor
        return current_solution
    def move_operator(self, devices, current_solution):
        # move an operator from u to a new v
        moved_op_ids = []
        # randomly select an operator from mappings
        counter = 0
        while counter >= 10:
            random_ms_id = random.randint(0, len(current_solution)-1)
            mapping = current_solution[random_ms_id]
            op_id = mapping[0]
            op_code = mapping[1]
            dev_id = mapping[2]
            op_load = mapping[3]
            filtered_dev_ids = [item for item in self.filter_devices(devices, (op_id, op_code, op_load)) if item != dev_id]
            if len(filtered_dev_ids) == 0:
                counter += 1
                continue
            random_dev_id = random.choice(filtered_dev_ids)
            new_neighbor = copy.deepcopy(current_solution)
            for mapping in new_neighbor:
                if mapping[0] == op_id:
                    mapping[2] = random_dev_id
            return new_neighbor
        return current_solution

    def change_operator(self, devices, current_solution):
        # change deployed operators to other compartible operator
        couter = 0
        while couter <= 10:
            random_ms_id = random.randint(0, len(current_solution) - 1)
            mapping = current_solution[random_ms_id]
            op_id = mapping[0]
            op_code = mapping[1]
            dev_id = mapping[2]
            dev_name = devices[dev_id]["model"]
            op_load = mapping[3]
            service_code = self.microservice_data["ms_types"][random_ms_id]
            # get peer operators that can hold the loads
            other_op_codes = [item for item in self.get_peer_operators(service_code, dev_name, op_load) if item != op_code]
            if len(other_op_codes) == 0:
                couter += 1
                continue
            old_resource_requirements = self.operator_profiles[op_code]["requirements"]["system"]
            old_resource_requirements["cpu"] = cpu_consumption(op_code, dev_name, op_load)
            # self.undeploy(devices, mapping)
            qualified_opcodes = []
            for other_op_code in other_op_codes:
                resource_requirements = self.operator_profiles[other_op_code]["requirements"]["system"]
                resource_requirements["cpu"] = max(0, cpu_consumption(other_op_code, dev_name, op_load) -
                                                   old_resource_requirements["cpu"])
                resource_requirements["memory"] = max(0,
                                                      resource_requirements["memory"] - old_resource_requirements["memory"])
                if self.is_system_consistent(devices[dev_id]["resources"]["system"], resource_requirements):
                    qualified_opcodes.append(other_op_code)
            if len(qualified_opcodes) == 0:
                couter += 1
                continue
            random_op_code = random.choice(qualified_opcodes)
            new_neighbor = copy.deepcopy(current_solution)
            for mapping in new_neighbor:
                if mapping[0] == op_id:
                    mapping[1] = random_op_code
            return new_neighbor
        return current_solution

    def merge_microservices(self, devices, current_solution):
        # merge the same type of microservices
        # for each microservice, try to reuse other's operator
        same_microservices = {}
        # categorize microservices
        for ms_id, mapping in enumerate(current_solution):
            op_id = mapping[0]
            service_code = self.microservice_data["ms_types"][ms_id]
            if service_code not in same_microservices.keys():
                same_microservices[service_code] = {}
            if op_id not in same_microservices[service_code].keys():
                same_microservices[service_code][op_id] = []
            same_microservices[service_code][op_id].append(ms_id)
        counter = 0
        while counter <= 10:
            random_ms_id = random.randint(0, len(current_solution) - 1)
            mapping = current_solution[random_ms_id]
            service_code = self.microservice_data["ms_types"][random_ms_id]
            wf_id = self.ms_to_wf[random_ms_id]
            rate = self.workflows[wf_id]["rate"]
            operator_ids = [item for item in same_microservices[service_code].keys() if item != mapping[0]]
            if len(operator_ids) == 0:
                counter += 1
                continue
            reusable_op_mappings = []
            for other_op_id in operator_ids:
                other_index = same_microservices[service_code][other_op_id][0]
                other_mapping = current_solution[other_index]
                if self.operator_reusable(devices, other_mapping, rate):
                    reusable_op_mappings.append(other_mapping)
            if len(reusable_op_mappings) == 0:
                counter += 1
                continue
            random_mapping = random.choice(reusable_op_mappings)
            selected_op_id = random_mapping[0]
            new_neighbor = copy.deepcopy(current_solution)
            new_neighbor[random_ms_id] = random_mapping
            for mapping in new_neighbor:
                if mapping[0] == selected_op_id:
                    mapping[3] += rate
            return new_neighbor
        return current_solution

    def consume_operators(self, devices, solution):
        deployed_op_ids = []
        for mapping in solution:
            if mapping[0] in deployed_op_ids:
                continue
            self.deploy(devices, mapping)
            deployed_op_ids.append(mapping[0])
        return devices

    def generate_new(self, current_device, current_solution):
        functions = [self.swap_resource, self.move_operator, self.change_operator, self.merge_microservices]
        # T higher, more purturbations
        T_diff = (self.T0 - self.Tf) / 4
        T1 = self.T0 + T_diff
        T2 = T1 + T_diff
        T3 = T2 + T_diff
        best_neighbor = current_solution
        best_utility = self.calculate_utility(current_solution)

        new_solution = current_solution
        if self.T >= self.T0 and self.T < T1:
            purturb_num = 4
        elif self.T >= T1 and self.T < T2:
            purturb_num = 3
        elif  self.T >= T2 and self.T < T3:
            purturb_num = 2
        else:
            purturb_num = 1
        selected_functions = random.sample(functions, purturb_num)
        for func in selected_functions:
            new_solution = func(current_device, current_solution)
            new_utility = self.calculate_utility(new_solution)
            if new_utility > best_utility:
                best_utility = new_utility
                best_neighbor = new_solution
        return best_neighbor

    def run(self):
        count = 0
        # annealing
        # current_devices = copy.deepcopy(self.devices)
        solution = self.initial_solution()
        # self.consume_operators(current_devices, solution)
        f_best = float("-inf")
        while self.T > self.Tf:
            # inner iteration
            for i in range(self.iter):
                f = self.calculate_utility(solution)
                current_devices = self.consume_operators(copy.deepcopy(self.devices), solution)
                new_solution = self.generate_new(current_devices, solution)
                f_new = self.calculate_utility(new_solution)
                if self.Metrospolis(f, f_new):
                    solution = new_solution
                    f_best = f_new

            self.history['f'].append(f_best)
            self.history['T'].append(self.T)
            #cooling
            self.T = self.T * self.alpha
            count += 1

        return solution, f_best

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
            wa = 0.05
            wb = 0.95
            if acc_max == acc_min:
                A = accuracy
            else:
                A = (accuracy - acc_min) / (acc_max - acc_min)
            B = (delay_tol - delay) / delay_tol
            utility = wa * A + wb * B
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

    def make_decision(self):
        print("Running Simulated Annealing decision maker")
        best_solution, best_utility = self.run()

        # # write the output into files
        # self.save_list_to_json(best_solution, "mock/solution.json")
        # np.save("mock/transmission.npy", self.transmission_matrix)
        # # self.save_list_to_json(self.transmission_matrix, "mock/solution.json")
        # self.save_dict_to_json(self.microservice_data, "mock/microservicedata.json")
        # self.save_list_to_json(self.devices, "mock/devices.json")
        # self.save_list_to_json(self.workflows, "mock/workflows.json")
        # self.save_list_to_json(self.operator_data, "mock/operatordata.json")

        return best_solution, best_utility


