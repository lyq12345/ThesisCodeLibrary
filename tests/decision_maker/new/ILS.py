import copy
import os
import json
from pathlib import Path
import sys
import numpy as np
sys.path.append(str(Path(__file__).resolve().parents[1]))
from status_tracker.rescons_models import cpu_consumption
from tests.decision_maker.new.Greedy_deploy import Greedy_decider
import random

cur_dir = os.getcwd()

speed_lookup_table = None
power_lookup_table = None
with open(os.path.join(cur_dir, "../status_tracker/speed_lookup_table.json"), 'r') as file:
    speed_lookup_table = json.load(file)

with open(os.path.join(cur_dir, "../status_tracker/power_lookup_table.json"), 'r') as file:
    power_lookup_table = json.load(file)

class Iterated_LS_decider:
    def __init__(self, workflows, microservice_data, operator_data, devices, operators, transmission_matrix, effective_time, iteration_time=20, max_no_improvement=10):
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

        self.effective_time = effective_time
        self.transmission_matrix = transmission_matrix
        self.AMax = []
        self.Amin = []
        self.calculate_max_min_acc(workflows)

        self.iteration_time = iteration_time
        self.max_no_improvement = max_no_improvement

    def get_peer_operators(self, service_code, dev_name, op_load):
        candidate_op_codes = []
        for op in self.operator_profiles:
            if op["object_code"] == service_code:
                if op_load <= 1/speed_lookup_table[op["id"]][dev_name]:
                    candidate_op_codes.append(op["id"])

        return candidate_op_codes

    def create_operator_graph(self, solution):
        operator_graph = {}
        ms_id = 0
        for wf_id, workflow in enumerate(self.workflows):
            microservices = workflow["workflow"]
            pre_op_id = -1
            for id, microservice in enumerate(microservices):
                mapping = solution[ms_id]
                if len(mapping) == 0:
                    ms_id += 1
                    continue
                op_id = mapping[0]
                if op_id not in operator_graph.keys():
                    operator_graph[op_id] = []
                if id == 0:
                    pre_op_id = op_id
                else:
                    operator_graph[op_id].append([pre_op_id, wf_id])
                    pre_op_id = op_id
                ms_id += 1
        return operator_graph

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
            if len(mapping) == 0:
                continue
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

    def swap_resource(self, devices, current_solution, neighbors):
        # replace an active device with a new resource; all operators on it should be moved
        ops_on_devices = [[] for _ in range(len(devices))]
        traversed_op_ids = []
        for mapping in current_solution:
            if len(mapping) == 0:
                continue
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
                    if len(mapping) == 0:
                        continue
                    if mapping[2] == dev_id:
                        new_neighbor[ms_id][2] = other_dev_id
                # TODO: effective remove repeatable?
                if new_neighbor not in neighbors:
                    neighbors.append(new_neighbor)

    def move_operator(self, devices, current_solution, neighbors):
        # move an operator from u to a new v
        moved_op_ids = []
        for mapping in current_solution:
            if len(mapping) == 0:
                continue
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
                    if len(mapping) == 0:
                        continue
                    if mapping[0] == op_id:
                        mapping[2] = other_dev_id
                # TODO: effective remove repeatable?
                if new_neighbor not in neighbors:
                    neighbors.append(new_neighbor)
            moved_op_ids.append(op_id)

    def change_operator(self, devices, current_solution, neighbors):
        # change deployed operators to other compatible operator
        changed_op_ids = []
        for ms_id, mapping in enumerate(current_solution):
            if len(mapping) == 0:
                continue
            # device_copy = copy.deepcopy(devices)
            op_id = mapping[0]
            if op_id in changed_op_ids:
                continue
            op_code = mapping[1]
            dev_id = mapping[2]
            dev_name = devices[dev_id]["model"]
            op_load = mapping[3]
            service_code = self.microservice_data["ms_types"][ms_id]
            # get peer operators that can hold the loads
            other_op_codes = [item for item in self.get_peer_operators(service_code, dev_name, op_load) if item != op_code]
            old_resource_requirements = self.operator_profiles[op_code]["requirements"]["system"]
            old_resource_requirements["cpu"] = cpu_consumption(op_code, dev_name, op_load)
            # self.undeploy(devices, mapping)
            for other_op_code in other_op_codes:
                resource_requirements = self.operator_profiles[other_op_code]["requirements"]["system"]
                resource_requirements["cpu"] = max(0, cpu_consumption(other_op_code, dev_name, op_load) - old_resource_requirements["cpu"])
                resource_requirements["memory"] = max(0, resource_requirements["memory"]-old_resource_requirements["memory"])

                if self.is_system_consistent(devices[dev_id]["resources"]["system"], resource_requirements):
                    new_neighbor = copy.deepcopy(current_solution)
                    for mapping in new_neighbor:
                        if len(mapping) == 0:
                            continue
                        if mapping[0] == op_id:
                            mapping[1] = other_op_code
                    if new_neighbor not in neighbors:
                        neighbors.append(new_neighbor)
            changed_op_ids.append(op_id)

    def perturbation(self, devices, current_solution):
        # merge the same type of microservices
        # for each microservice, try to reuse other's operator
        same_microservices = {}
        # categorize microservices
        for ms_id, mapping in enumerate(current_solution):
            if len(mapping) == 0:
                continue
            op_id = mapping[0]
            service_code = self.microservice_data["ms_types"][ms_id]
            if service_code not in same_microservices.keys():
                same_microservices[service_code] = {}
            if op_id not in same_microservices[service_code].keys():
                same_microservices[service_code][op_id] = []
            same_microservices[service_code][op_id].append(ms_id)
        counter = 0

        # "op_id": ["pre1", "pre2", "pre3"]
        operator_graph = self.create_operator_graph(current_solution)

        changable_op_ids = []
        for code in same_microservices.keys():
            op_ids = same_microservices[code].keys()
            if len(op_ids) > 0:
                for op_id in op_ids:
                    changable_op_ids.append(op_id)
        changable_edges = []
        for op_id in changable_op_ids:
            for pre_op_id, wf_id in operator_graph[op_id]:
                changable_edges.append([pre_op_id, op_id, wf_id])

        if len(changable_edges) == 0:
            return current_solution

        while counter <= 3:
            random_edge = random.choice(changable_edges)
            from_op_id = random_edge[0]
            to_op_id = random_edge[1]
            to_op_code = self.operator_data[from_op_id]
            to_service_code = self.operator_profiles[to_op_code]["object_code"]
            selected_wf_id = random_edge[2]
            rate = self.workflows[selected_wf_id]["rate"]
            other_op_ids = [item for item in same_microservices[to_service_code].keys() if item != to_op_id]
            if len(other_op_ids) == 0:
                counter += 1
                continue
            new_op_id = random.choice(other_op_ids)
            new_op_mapping = None
            to_op_mapping = None
            for mapping in current_solution:
                if len(mapping) == 0:
                    continue
                if mapping[0] == new_op_id:
                    new_op_mapping = mapping
                if mapping[0] == to_op_id:
                    to_op_mapping = mapping

            if to_op_mapping[3] - rate > 0:
                self.change_operator_load(devices, to_op_id, to_op_mapping[2], to_op_mapping[3], to_op_mapping[3] - rate)
            else:
                self.undeploy(devices, to_op_mapping)

            if not self.operator_reusable(devices, new_op_mapping, rate):
                counter += 1
                if to_op_mapping[3] - rate > 0:
                    self.change_operator_load(devices, to_op_id, to_op_mapping[2], to_op_mapping[3] - rate, to_op_mapping[3])
                else:
                    self.deploy(devices, to_op_mapping)
                continue

            # 3. change solution ()
            for id, mapping in enumerate(current_solution):
                if len(mapping) == 0:
                    continue
                if mapping[0] == new_op_id:
                    mapping[3] += rate
                if mapping[0] == to_op_id:
                    if self.ms_to_wf[id] == selected_wf_id:
                        temp_new_mapping = new_op_mapping
                        temp_new_mapping[3] += rate
                        current_solution[id] = temp_new_mapping
                    else:
                        mapping[3] -= rate
            return current_solution

        return current_solution
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
            self.change_operator(devices, current_solution, neighbors)

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
            # TODO: this cause resource bugs
            if best_neighbor_utility > best_utility:
                best_solution = best_neighbor
                best_utility = best_neighbor_utility
            # best_neighbor, best_neighbor_utility = self.improve_solution(best_solution, 4)
            # if best_neighbor_utility > best_utility:
            #     best_solution = best_neighbor
            #     best_utility = best_neighbor_utility

            if best_utility <= current_best_utility: # if no improvements made:
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
            unsatisfied = False

            for i in range(len(workflow["workflow"])):
                mapping = solution[ms_id]
                if len(mapping) == 0:
                    unsatisfied = True
                    break
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
            if unsatisfied:
                continue
            wa = 0.05
            wb = 0.95
            if acc_max == acc_min:
                A = accuracy
            else:
                A = (accuracy-acc_min)/(acc_max-acc_min)
            B = (delay_tol - delay) / delay_tol
            utility = wa*A + wb*B
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

    def change_operator_load(self, devices, op_id, dev_id, pre_rate, new_rate):
        op_code = self.operator_data[op_id]
        dev_name =devices[dev_id]["model"]
        resource_requirements = self.operator_profiles[op_code]["requirements"]["system"]
        cpu_difference = cpu_consumption(op_code, dev_name, new_rate) - cpu_consumption(op_code, dev_name, pre_rate)
        devices[dev_id]["resources"]["system"]["cpu"] += cpu_difference

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
            if len(mapping) == 0:
                continue
            if mapping[0] in deployed_op_ids:
                continue
            self.deploy(devices, mapping)
            deployed_op_ids.append(mapping[0])
    def make_decision(self):
        print("Running Iterated Local Search decision maker")
        counter = 0
        X = self.initial_solution()
        f = self.calculate_utility(X)

        while counter <= self.max_no_improvement:
            # consume the resources
            device_copy = copy.deepcopy(self.devices)
            self.consume_operators(device_copy, X)

            X1, f1 = self.local_search(device_copy, X)
            if f1 > f:
                X = X1
                f = f1
                counter = 0
            else:
                counter += 1
                X = self.perturbation(device_copy, X)

        # write the output into files
        # self.save_list_to_json(best_solution, "mock/solution.json")
        # np.save("mock/transmission.npy", self.transmission_matrix)
        # # self.save_list_to_json(self.transmission_matrix, "mock/solution.json")
        # self.save_dict_to_json(self.microservice_data, "mock/microservicedata.json")
        # self.save_list_to_json(self.devices, "mock/devices.json")
        # self.save_list_to_json(self.workflows, "mock/workflows.json")
        # self.save_list_to_json(self.operator_data, "mock/operatordata.json")

        return X, f


