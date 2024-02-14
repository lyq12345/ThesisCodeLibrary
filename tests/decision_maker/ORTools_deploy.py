import numpy as np
import json
import os
from ortools.linear_solver import pywraplp

cur_dir = os.getcwd()

speed_lookup_table = None
power_lookup_table = None
with open(os.path.join(cur_dir, "../status_tracker/speed_lookup_table.json"), 'r') as file:
    speed_lookup_table = json.load(file)

with open(os.path.join(cur_dir, "../status_tracker/power_lookup_table.json"), 'r') as file:
    power_lookup_table = json.load(file)
class ORTools_Decider:
    def __init__(self, tasks, devices, operators, transmission_matrix):
        self.num_tasks = len(tasks)
        self.num_devices = None
        self.operator_list = operators
        self.tasks = tasks
        self.device_data = self.create_device_model(tasks, devices, transmission_matrix)
        self.operator_data = self.create_operator_model(tasks, devices, operators)

    def create_device_model(self, tasks, devices, transmission_matrix):
        data = {}
        """Stores the data for the problem."""
        data["resource_capability"] = []
        data["data_sources"] = []
        data["device_models"] = []

        for device in devices:
            data["resource_capability"].append([device["resources"]["system"][key] for key in device["resources"]["system"]])
            data["device_models"].append(device["model"])

        for task in tasks:
            for device in devices:
                if device["resources"]["hardware"] is not None:
                    for hardware in device["resources"]["hardware"]:
                        if hardware["id"] == task["source"]:
                            data["data_sources"].append(device["id"])


        data["transmission_speed"] = transmission_matrix

        self.num_devices = len(devices)
        return data

    def create_operator_model(self, tasks, devices, operators):
        data = {}
        data["operator_accuracies"] = []
        data["resource_requirements"] = []
        data["processing_speed"] = []
        data["power_consumptions"] = []
        data["operator_types"] = []
        data["operator_ids"] = []
        count = 0
        for task in tasks:
            object_type = task["object"]
            for op in operators:
                if op["type"] == "processing" and op["object"] == object_type:
                    data["operator_accuracies"].append(op["accuracy"])
                    data["resource_requirements"].append([op["requirements"]["system"][key] for key in op["requirements"]["system"]])
                    data["operator_types"].append(op["object_code"])
                    data["operator_ids"].append(op["id"])
                    # data["processing_speed"].append([speed_lookup_table[op_name][dev_name] for dev_name in device_names])
                    # data["power_consumptions"].append([speed_lookup_table[op_name][dev_name] for dev_name in device_names])
                    count += 1
        self.num_operators = count

        return data

    def make_decision(self):
        print("Running ORTools decision maker")
        # Create the SCIP solver
        solver = pywraplp.Solver.CreateSolver('SCIP')

        T = self.num_tasks
        O = self.num_operators
        D = self.num_devices

        # Create binary decision variables x_ij for the binary matrix x
        x = {}  # x_ijk - task i transmit to operator j on device k
        for i in range(T):
            for j in range(O):
                for k in range(D):
                    x[i, j, k] = solver.IntVar(0, 1, f'x_{i}_{j}_{k}')

        # Each task need to map to one operator
        for i in range(T):
            solver.Add(solver.Sum([x[i,j,k] for j in range(O) for k in range(D)]) == 1)

        # the operator type should be consistent with tasks req
        for i in range(T):
            for j in range(O):
                for k in range(D):
                    solver.Add(x[i, j, k]*self.tasks[i]["object_code"] == x[i, j, k]*self.operator_data["operator_types"][j])

        # Each operator is deployed to at most 1 device.
        for j in range(O):
            for i in range(T):
                solver.Add(solver.Sum([x[i, j, k] for k in range(D)]) <= 1)

        # operator requirement sum in each device should not exceed its capacity
        for k in range(D):
            for t in range(4):
                solver.Add(solver.Sum([x[i, j, k] * self.operator_data["resource_requirements"][j][t] for j in range(O) for i in range(T)]) <=
                       self.device_data["resource_capability"][k][t])

        # operator rate sum should not exceed operator processing rate
        for j in range(O):
            for k in range(D):
                dev_name = self.device_data["device_models"][k]
                solver.Add(solver.Sum([x[i,j,k]*self.tasks[i]["rate"] for i in range(T)]) <= 1/speed_lookup_table[self.operator_data["operator_ids"][j]][dev_name])
        # for j in range(O):
        #     # find all sources (i) that transmit to j
        #     sum_j = []
        #     processing_rate = 0
        #     for i in range(T):
        #         for k in range(D):
        #             rate = self.tasks[i]["rate"]
        #             dev_name = self.device_data["device_models"][k]
        #             op_id = self.operator_data["operator_ids"][j]
        #             processing_rate += x[i,j,k] * (1 / speed_lookup_table[op_id][dev_name])
        #             sum_j.append(x[i,j,k] * rate)
        #     solver.Add(solver.Sum(sum_j) <= processing_rate)

        utilities = []
        for i in range(T):
            for j in range(O):
                for k in range(D):
                    source_device_id = self.device_data["data_sources"][i]
                    op_id = self.operator_data["operator_ids"][j]
                    delay_tol = self.tasks[i]["delay"]
                    transmission_delay = self.device_data["transmission_speed"][source_device_id][k]
                    dev_name = self.device_data["device_models"][k]
                    processing_delay = speed_lookup_table[op_id][dev_name]
                    utilities.append(x[i, j, k] * (self.operator_data["operator_accuracies"][j]
                                     - max(0, 1 - delay_tol/(processing_delay+transmission_delay))))

        solver.Maximize(solver.Sum(utilities))

        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL:
            solution = [None]*T
            best_utility = solver.Objective().Value()
            for i in range(T):
                for j in range(O):
                    for k in range(D):
                        if x[i, j, k].solution_value() != 0:
                            op_id = self.operator_data["operator_ids"][j]
                            solution[i] = (j, op_id, k)
                            # print(f"data flow {data_flow_count}: ")
                            # source_device_id = self.device_data["data_sources"][i]
                            # print(f"sensor on device {source_device_id} transmits data to operator {j} deployed on device {k}")
                            # accuracy = self.operator_data["operator_accuracies"][j]
                            # delay = self.device_data["transmission_speed"][source_device_id][k] + self.operator_data["processing_speed"][j][k]
                            # print(f"accuracy: {accuracy}")
                            # print(f"delay: {delay}")
                            #
                            # print("------------------------------------------------------------")
                            # data_flow_count += 1
            return solution, best_utility

        else:
            print("No optimal solution found.")
            return None



