import numpy as np
import math
import copy
import random


speed_lookup_table = {
  0: {
    "jetson-nano": 0.5549,
    "raspberrypi-4b": 1.0702,
    "jetson-xavier": 0.4276
  },
  1: {
        "jetson-nano": 4.364,
        "raspberrypi-4b": 7.0823,
        "jetson-xavier": 2.6235
    },
  2: {
    "jetson-nano": 0.5864,
    "raspberrypi-4b": 1.0913,
    "jetson-xavier": 0.4605
  },
  3: {
    "jetson-nano": 4.4829,
    "raspberrypi-4b": 7.2191,
    "jetson-xavier": 3.8648
  }
}

power_lookup_table = {
  "joelee0515/firedetection:yolov3-measure-time": {
    "jetson-nano": 2916.43,
    "raspberrypi-4b": 1684.4,
    "jetson-xavier": 1523.94
  },
  "joelee0515/firedetection:tinyyolov3-measure-time": {
    "jetson-nano": 1584.53,
    "raspberrypi-4b": 1174.39,
    "jetson-xavier": 780.97
  },
  "joelee0515/humandetection:yolov3-measure-time": {
    "jetson-nano": 2900.08,
    "raspberrypi-4b": 1694.41,
    "jetson-xavier": 1540.61
  },
  "joelee0515/humandetection:tinyyolov3-measure-time": {
    "jetson-nano": 1191.19,
    "raspberrypi-4b": 1168.31,
    "jetson-xavier": 803.95
  }
}

class Greedy_deploy:
    def __init__(self, tasks, devices, operators, transmission_matrix):
        self.tasks = tasks
        self.devices = devices
        self.operators = operators
        self.transmission_matrix = transmission_matrix

        self.solution = [None]*len(tasks)

    def get_candidate_operators(self, task):
        object = task["object"]
        candidate_operators = []
        for op in self.operators:
            if op["object"] == object:
                candidate_operators.append(op)
        candidate_op_ids = [d["id"] for d in candidate_operators]
        return candidate_op_ids


    def initial_solution(self):
        devices_copy = copy.deepcopy(self.devices)
        for task in self.tasks:
            task_id = task["id"]
            candidate_op_ids = self.get_candidate_operators(task)
            selected_op_id = random.choice(candidate_op_ids)
            candidate_device_ids = self.filter_devices(devices_copy, selected_op_id)
            selected_device_id = random.choice(candidate_device_ids)
            self.deploy(devices_copy, (selected_op_id, selected_device_id))
            self.solution[task_id] = (selected_op_id, selected_device_id)


    def perturbation(self, current_solution):
        # randomly swap two task assignments of the same type
        # categorize tasks with the same object type
        type_dict = {}

        for i, mapping in enumerate(current_solution):
            operator_id = mapping[0]
            object_type = self.operators[operator_id]["object"]
            if object_type not in type_dict.keys():
                type_dict[object_type] = []
            type_dict[object_type].append(i)
        swapable_objects = []
        for key, value in type_dict.items():
            if len(value) >= 2:
                swapable_objects.append(key)
        if len(swapable_objects) == 0:
            return current_solution
        new_solution = current_solution[:]
        selected_group = random.choice(swapable_objects)
        task_ids = type_dict[selected_group]
        swap_id1, swap_id2 = random.sample(task_ids, 2)
        new_solution[swap_id1], new_solution[swap_id2] = new_solution[swap_id2], new_solution[swap_id1]
        return new_solution






    def get_neighbors(self, current_solution, tabu_list):
        # moving one operator to another device; change to another operator
        device_copy = copy.deepcopy(self.devices)
        # consume the devices
        for mapping in current_solution:
            self.deploy(device_copy, mapping)
        # print("current solution: ", current_solution)

        neighbors = []
        for i in range(len(current_solution)):
            candidate_ops = self.get_candidate_operators(self.tasks[i])
            mapping = current_solution[i]

            # undeploy to release the resources:
            self.undeploy(device_copy, mapping)
            # for all other candidate operators:
            for op_id in candidate_ops:
                # move operator to a different device
                filtered_devices = self.filter_devices(device_copy, op_id)
                for dev_id in filtered_devices:
                    if op_id == mapping[0] and dev_id==mapping[1]:
                        continue
                    neighbor = current_solution[:]
                    neighbor[i] = (op_id, dev_id)
                    if neighbor not in tabu_list:
                        neighbors.append(neighbor)
                    # print(neighbor)
        return neighbors

    def tabu_search(self, initial_solution, max_iterations, tabu_list_size, max_no_improvements):
        best_solution = initial_solution
        best_utility = self.calculate_utility(best_solution)
        current_solution = initial_solution
        tabu_list = []
        count = 0

        for i in range(max_iterations):

            current_solution = self.local_search(current_solution, tabu_list)
            current_utility = self.calculate_utility(current_solution)

            # if there's improvements
            if current_utility > best_utility:
                best_solution = current_solution
                best_utility = current_utility
                tabu_list.append(current_solution)
                if len(tabu_list) > tabu_list_size:
                    tabu_list.pop(0)
                # print(f"best solution after iteration {i}: {best_utility}")
            else:
                if count <= max_no_improvements:
                    # perturbation
                    current_solution = self.perturbation(current_solution)
                    count += 1
                    # print("No better solution found. Perturbation performed.")
                else:
                    break
        # print(best_solution, self.calculate_utility(best_solution))

        return best_solution, best_utility





    def local_search(self, current_solution, tabu_list):
        neighbors = self.get_neighbors(current_solution, tabu_list)
        best_neighbor = None
        best_neighbor_utility = float('-inf')

        for neighbor in neighbors:
            # check list
            if neighbor not in tabu_list:
                neighbor_utility = self.calculate_utility(neighbor)
                if neighbor_utility > best_neighbor_utility:
                    best_neighbor = neighbor
                    best_neighbor_utility = neighbor_utility

        return best_neighbor





    def iterated_local_search(self, max_iterations, max_no_improve):
        # get initial solution
        self.initial_solution()
        # calculate the cost for initial solutio
        initial_cost = self.calculate_utility(self.solution)
        self.get_neighbors(self.solution)

        # for i in range(max_iterations):
        #     # perturbation
        #     self.local_search()

            # if find a better solution
                #assign best solution


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

    def filter_devices(self, devices, operator_id):
        filtered_devices = []
        operator = self.operators[operator_id]
        for dev in devices:
            if self.is_system_consistent(dev["resources"]["system"], operator["requirements"]["system"]):
                filtered_devices.append(dev)
        filtered_device_ids = [d["id"] for d in filtered_devices]
        return filtered_device_ids

    def calculate_utility(self, solution):
        sum_uti = 0
        for task_id, mapping in enumerate(solution):
            source_device_id = self.tasks[task_id]["source"]
            operator_id = mapping[0]
            device_id = mapping[1]
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
        operator_id = mapping[0]
        device_id = mapping[1]
        operator_resource = {}
        for op in self.operators:
            if operator_id == op["id"]:
                operator_resource = op["requirements"]["system"]

        for type, amount in operator_resource.items():
            devices[device_id]["resources"]["system"][type] -= amount

    def undeploy(self, devices, mapping):
        operator_id = mapping[0]
        device_id = mapping[1]
        operator_resource = {}
        for op in self.operators:
            if operator_id == op["id"]:
                operator_resource = op["requirements"]["system"]

        for type, amount in operator_resource.items():
            devices[device_id]["resources"]["system"][type] += amount

    def make_decision(self):
        print("Running Local Search decision maker")
        # self.iterated_local_search(max_iterations=10, max_no_improve=5)
        self.initial_solution()
        best_solution, best_utility = self.tabu_search(self.solution, max_iterations=100, tabu_list_size=20, max_no_improvements=20)
        return best_solution, best_utility


