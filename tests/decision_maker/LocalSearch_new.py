import copy
import os
from TOPSIS_deploy import TOPSIS_decider
import json

cur_dir = os.getcwd()

speed_lookup_table = None
power_lookup_table = None
with open(os.path.join(cur_dir, "../status_tracker/speed_lookup_table.json"), 'r') as file:
    speed_lookup_table = json.load(file)

with open(os.path.join(cur_dir, "../status_tracker/power_lookup_table.json"), 'r') as file:
    power_lookup_table = json.load(file)

class LocalSearch_new:
    def __init__(self, tasks, devices, operators, transmission_matrix):
        self.tasks = tasks
        self.devices = copy.deepcopy(devices)
        self.operators = operators
        self.transmission_matrix = transmission_matrix

    def get_candidate_operators(self, task_id):
        object = self.tasks[task_id]["object"]
        candidate_operators = []
        for op in self.operators:
            if op["object"] == object:
                candidate_operators.append(op)
        candidate_op_ids = [d["id"] for d in candidate_operators]
        return candidate_op_ids
    def calculate_resource_consumption(self, solution):
        cpu_consumptions = [0]*len(self.devices)
        ram_consumptions = [0]*len(self.devices)
        for i, mapping in enumerate(solution):
            op_id = mapping[1]
            dev_id = mapping[2]
            op_resource = self.operators[op_id]["requirements"]["system"]
            cpu_consumptions[dev_id] += op_resource["cpu"]
            ram_consumptions[dev_id] += op_resource["memory"]
        for i in range(len(cpu_consumptions)):
            cpu_consumptions[i] = cpu_consumptions[i] / self.devices[i]["resources"]["system"]["cpu"]
            ram_consumptions[i] = ram_consumptions[i] / self.devices[i]["resources"]["system"]["memory"]
        print("CPU consumptions: ")
        print(cpu_consumptions)
        print("Memory consumptions:")
        print(ram_consumptions)

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

        topsis_decider = TOPSIS_decider(self.tasks, self.devices, self.operators, self.transmission_matrix)
        init_solution, init_utility = topsis_decider.make_decision(display=False)
        self.calculate_resource_consumption(init_solution)
        return init_solution

    def swap_resource(self, devices, current_solution, neighbors):
        # replace an active device with a new resource; all operators on it should be moved
        ops_on_devices = [[] for _ in range(len(devices))]
        for task_id, mapping in enumerate(current_solution):
            op_global_id = mapping[0]
            op_id = mapping[1]
            dev_id = mapping[2]
            ops_on_devices[dev_id].append([op_global_id, op_id])
        for dev_id, ops in enumerate(ops_on_devices):
            # find other devices that can accommodate all ops on dev_id
            op_ids = [item[1] for item in ops]
            filtered_dev_ids = [item for item in self.filter_devices_multiop(devices, op_ids) if item != dev_id]
            for other_dev_id in filtered_dev_ids:
                new_neighbor = copy.deepcopy(current_solution)
                for task_id, mapping in enumerate(new_neighbor):
                    if mapping[2] == dev_id:
                        new_neighbor[task_id][2] = other_dev_id
                # TODO: effective remove repeatable?
                if new_neighbor not in neighbors:
                    neighbors.append(new_neighbor)

    def move_operator(self, devices, current_solution, neighbors):
        # move an operator from u to a new v
        for task_id, mapping in enumerate(current_solution):
            op_global_id = mapping[0]
            op_id = mapping[1]
            dev_id = mapping[2]
            filtered_dev_ids = [item for item in self.filter_devices(devices, op_id) if item != dev_id]
            for other_dev_id in filtered_dev_ids:
                new_neighbor = copy.deepcopy(current_solution)
                new_neighbor[task_id][2] = other_dev_id
                # TODO: effective remove repeatable?
                if new_neighbor not in neighbors:
                    neighbors.append(new_neighbor)

    def change_operator(self, devices, current_solution, neighbors):
        # change deployed operators to other compartible operator
        for task_id, mapping in enumerate(current_solution):
            op_id = mapping[1]
            dev_id = mapping[2]
            other_op_ids = [item for item in self.get_candidate_operators(task_id) if item != op_id]
            self.undeploy(devices, mapping)
            for other_op_id in other_op_ids:
                if self.is_system_consistent(devices[dev_id]["resources"]["system"], self.operators[other_op_id]["requirements"]["system"]):
                    new_neighbor = copy.deepcopy(current_solution)
                    new_neighbor[task_id][1] = other_op_id
                    if new_neighbor not in neighbors:
                        neighbors.append(new_neighbor)
            self.deploy(devices, mapping)

    def merge_request(self):
        pass


    def get_neighbors(self, current_solution):
        """
        exploration strategies:
        1) swap resources: replace an active device u with a new one v
        2) relocate operator: relocate a single operator i from its location u to a new device v
        3) change operator: change operator i on a device u to operator j that does the same thing
        4) merge tasks with same type of requests
        """
        # moving one operator to another device; change to another operator
        device_copy = copy.deepcopy(self.devices)
        # consume the devices
        for mapping in current_solution:
            self.deploy(device_copy, mapping)
        # print("current solution: ", current_solution)

        neighbors = []
        self.swap_resource(device_copy, current_solution, neighbors)
        self.move_operator(device_copy, current_solution, neighbors)
        self.change_operator(device_copy, current_solution, neighbors)

        return neighbors

    def local_search(self):
        best_solution = self.initial_solution()
        best_utility = self.calculate_utility(best_solution)
        while True:
            best_neighbor_utility = best_utility
            best_neighbor = best_solution
            neighbors = self.get_neighbors(best_solution)
            # find the best neighbor
            for neighbor in neighbors:
                neighbor_utility = self.calculate_utility(neighbor)
                if neighbor_utility > best_neighbor_utility:
                    # self.calculate_resource_consumption(neighbor)
                    best_neighbor_utility = neighbor_utility
                    best_neighbor = neighbor
            if best_neighbor_utility > best_utility:
                best_solution = best_neighbor
                best_utility = best_neighbor_utility
            else:      # if no improvements made:
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
            requirements = self.operators[op_id]["requirements"]["system"]
            if not self.is_system_consistent(resources, requirements):
                return False
            else:
                self.deploy(device_copy, mapping)
        return True


    def filter_devices(self, devices, operator_id):
        filtered_devices = []
        operator = self.operators[operator_id]
        for dev in devices:
            if self.is_system_consistent(dev["resources"]["system"], operator["requirements"]["system"]):
                filtered_devices.append(dev)
        filtered_device_ids = [d["id"] for d in filtered_devices]
        return filtered_device_ids

    def filter_devices_multiop(self, devices, operator_ids):
        filtered_devices = []
        for dev in devices:
            accumulated_resources = {
                "cpu": 0,
                 "gpu": 0,
                "storage": 0,
                "memory": 0
            }
            for op_id in operator_ids:
                operator = self.operators[op_id]
                for key, value in operator["requirements"]["system"].items():
                    accumulated_resources[key] += value
            if self.is_system_consistent(dev["resources"]["system"], accumulated_resources):
                filtered_devices.append(dev)
        filtered_device_ids = [d["id"] for d in filtered_devices]
        return filtered_device_ids

    def calculate_utility(self, solution):
        sum_uti = 0
        for task_id, mapping in enumerate(solution):
            source_device_id = self.tasks[task_id]["source"]
            operator_id = mapping[1]
            device_id = mapping[2]
            accuracy = self.operators[operator_id]["accuracy"]
            delay = self.calculate_delay(operator_id, source_device_id, device_id)
            task_del = self.tasks[task_id]["delay"]
            utility = accuracy - max(0, (delay - task_del)/delay)
            sum_uti += utility
        cost = sum_uti
        return cost


    def calculate_delay(self, operator_id, source_device_id, device_id):
        device_model = self.devices[device_id]["model"]
        transmission_delay = self.transmission_matrix[source_device_id, device_id]
        processing_delay = speed_lookup_table[operator_id][device_model]
        return transmission_delay + processing_delay

    def calculate_power(self, operator, device_id):
        operator_name = operator["name"]
        device_model = self.devices[device_id]["model"]
        power = power_lookup_table[operator_name][device_model]
        return power

    def deploy(self, devices, mapping):
        operator_id = mapping[1]
        device_id = mapping[2]
        operator_resource = {}
        for op in self.operators:
            if operator_id == op["id"]:
                operator_resource = op["requirements"]["system"]

        for type, amount in operator_resource.items():
            devices[device_id]["resources"]["system"][type] -= amount

    def undeploy(self, devices, mapping):
        operator_id = mapping[1]
        device_id = mapping[2]
        operator_resource = {}
        for op in self.operators:
            if operator_id == op["id"]:
                operator_resource = op["requirements"]["system"]

        for type, amount in operator_resource.items():
            devices[device_id]["resources"]["system"][type] += amount

    def make_decision(self):
        print("Running Local Search New decision maker")
        best_solution, best_utility = self.local_search()
        return best_solution, best_utility


