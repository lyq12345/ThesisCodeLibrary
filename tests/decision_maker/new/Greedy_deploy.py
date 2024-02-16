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
            3. Reuse first
        until it's full; 
"""
class Greedy_decider:
    def __init__(self, workflows, devices, operators, transmission_matrix):
        self.workflows = workflows
        self.microservices_graph = None
        self.devices = copy.deepcopy(devices)
        self.operators = operators

        self.transmission_matrix = transmission_matrix
        self.create_microservice_model()

    def create_microservice_model(self):
        ms_num = sum([len(item["workflow"]) for item in self.workflows])
        self.microservices_graph = [[0 for _ in range(ms_num)] for _ in range(ms_num)]
        for wf_id, workflow in enumerate(self.workflows):
            microservices = workflow["workflow"]
            for i in range(len(microservices)-1):
                self.microservices_graph[microservices[i]][microservices[i+1]] = 1

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
        operator = self.operators[operator_id]
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
            accuracy = self.operators[operator_id]["accuracy"]
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
        operator_id = mapping[0]
        device_id = mapping[1]
        operator_resource = {}
        for op in self.operators:
            if operator_id == op["id"]:
                operator_resource = op["requirements"]["system"]

        for type, amount in operator_resource.items():
            devices[device_id]["resources"]["system"][type] -= amount

    def dijkstra(self, transmission_matrix, start, length):
        # 初始化距离数组，将起点距离设为0，其他点设为无穷大
        distances = {node: float('inf') for node in range(len(transmission_matrix))}
        distances[start] = 0

        # 创建优先队列，用于存储待处理的节点和其距离
        pq = [(0, start)]

        while pq:
            current_distance, current_node = heapq.heappop(pq)

            # 如果当前节点已经被处理，则继续下一个节点
            if current_distance > distances[current_node]:
                continue

            # 遍历当前节点的邻居节点
            for neighbor, weight in enumerate(transmission_matrix[current_node]):
                distance = current_distance + weight

                # 如果到达邻居节点的距离比已知的距离小，则更新距离并将邻居节点加入优先队列
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(pq, (distance, neighbor))

        # 找到路径长度小于等于给定长度的节点
        reachable_nodes = [node for node, dist in distances.items() if dist <= length]

        return reachable_nodes


    def make_decision(self, display=True):
        if display:
            print("Running Greedy decision maker")
        solution = []
        for workflow in self.workflows:
            source_device_id = workflow["source"]
            print("source", source_device_id)
            microservices = workflow["workflow"]
            print("microservices:", microservices)
            dev_path = self.dijkstra(self.transmission_matrix, source_device_id, len(microservices))
            # print(microservices)
            print("path", dev_path)

        utility = self.calculate_utility(solution)
        return solution, utility




