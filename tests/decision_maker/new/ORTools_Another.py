import numpy as np
import os
import json
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model
import collections
cur_dir = os.getcwd()

speed_lookup_table = None
power_lookup_table = None
with open(os.path.join(cur_dir, "../status_tracker/speed_lookup_table.json"), 'r') as file:
    speed_lookup_table = json.load(file)

with open(os.path.join(cur_dir, "../status_tracker/power_lookup_table.json"), 'r') as file:
    power_lookup_table = json.load(file)

class ORTools_Decider:
    def __init__(self, workflows, devices, operators, transmission_matrix, bandwidth_matrix=None):
        self.num_workflows = len(workflows)
        self.num_devices = None
        self.operator_list = operators
        self.workflows = workflows
        self.wf_ms_mapping = [[] for _ in range(len(workflows))]
        self.microservices_data = self.create_microservice_model(workflows)
        # print(self.wf_ms_mapping)
        # print(self.microservices_data)
        self.device_data = self.create_device_model(workflows, devices, transmission_matrix)
        self.operator_data = self.create_operator_model(operators)
        # print(self.device_data)



    def create_microservice_model(self, workflows):
        data = {"microservice_types": [], "microservice_rates": []}
        msid_count = 0
        for wf_id, workflow in enumerate(workflows):
            microservices = workflow["workflow"]
            rate = workflow["rate"]
            for microservice_type in microservices:
                self.wf_ms_mapping[wf_id].append(msid_count)
                data["microservice_types"].append(microservice_type)
                data["microservice_rates"].append(rate)
                msid_count += 1
        return data
    def create_device_model(self, workflows, devices, transmission_matrix):
        data = {}
        data["resource_capability"] = []
        data["data_sources"] = []
        data["device_models"] = []

        for device in devices:
            data["resource_capability"].append([device["resources"]["system"][key] for key in device["resources"]["system"]])
            data["device_models"].append(device["model"])

        for workflow in workflows:
            for device in devices:
                if device["resources"]["hardware"] is not None:
                    for hardware in device["resources"]["hardware"]:
                        if hardware["id"] == workflow["source"]:
                            data["data_sources"].append(device["id"])


        data["transmission_speed"] = transmission_matrix

        self.num_devices = len(devices)
        return data

    def create_operator_model(self, operators):
        data = {}
        data["operator_microservices"] = []
        data["operator_ids"] = []
        # data["operator_accuracies"] = []
        # data["resource_requirements"] = []
        # data["op_candidates"] = []
        count = 0
        for microservice in self.microservices_data["microservice_types"]:

            # candidate_ops = []
            for op in operators:
                if op["object_code"] == microservice:
                    data["operator_microservices"].append(microservice)
                    data["operator_ids"].append(op["id"])
                    # data["operator_ids"].append(op["id"])
                    # data["operator_codes"].append(op["object_code"])
                    # data["operator_accuracies"].append(op["accuracy"])
                    # data["resource_requirements"].append([op["requirements"]["system"][key] for key in op["requirements"]["system"]])
                    # data["processing_speed"].append([speed_lookup_table[op_name][dev_name] for dev_name in device_names])
                    # data["power_consumptions"].append([speed_lookup_table[op_name][dev_name] for dev_name in device_names])
                    count += 1
            # data["op_candidates"].append(candidate_ops)
        self.num_operators = count
        return data

    def make_decision(self):
        print("Running ORTools decision maker")
        # Create the SCIP solver
        model = cp_model.CpModel()

        microservice_deploy = collections.namedtuple("microservice_deploy", "operator device")


        W = len(self.workflows)
        M = len(self.microservices_data["microservice_types"])
        O = self.num_operators
        D = self.num_devices

        all_microservices = {}

        for wf_id, workflow in enumerate(self.workflows):
            for ms_id, ms in enumerate(workflow):
                suffix = f"_{wf_id, ms_id}"
                operator_var = model.NewIntVar(0, O, "operator"+suffix)
                device_var = model.NewIntVar(0, D, "device"+suffix)
                all_microservices[wf_id, ms_id] = microservice_deploy(operator=operator_var, device=device_var)

        # every microservice need to map one operator (two ms can map to 1)
        for i in range(M):
            solver.Add(solver.Sum([x[i,j,k] for j in range(O) for k in range(D)]) == 1)

        # assigned operator should be consistent with microservice
        for i in range(M):
            for j in range(O):
                for k in range(D):
                    solver.Add(x[i, j, k] * self.microservices_data["microservice_types"][i] == x[i, j, k] *
                               self.operator_data["operator_microservices"][j])

        # Each operator is deployed to at most 1 device.
        for j in range(O):
            for i in range(M):
                solver.Add(solver.Sum([x[i, j, k] for k in range(D)]) <= 1)

        resources = ["cpu", "gpu", "storage", "memory"]
        # operator requirement sum in each device should not exceed its capacity
        for k in range(D):
            for resource_id, t in enumerate(resources):
                solver.Add(solver.Sum([x[i, j, k] * self.operator_list[self.operator_data["operator_ids"][j]]["requirements"]["system"][t] for j in range(O) for i in range(M)]) <=
                       self.device_data["resource_capability"][k][resource_id])

        # operator rate sum should not exceed operator processing rate

        for j in range(O):
            # find the microservices that are mapped to operator j
            for k in range(D):
                solver.Add(solver.Sum([x[i,j,k]*self.microservices_data["microservice_rates"][i] for i in range(M)]) <= 1 / speed_lookup_table[self.operator_data["operator_ids"][j]][self.device_data["device_models"][k]])

        objectives = []
        for wf_id, workflow in enumerate(self.wf_ms_mapping):
            workflow_latencies = []
            accuracies = []
            delay_tol = self.workflows[wf_id]["delay"]
            for idx in range(len(workflow)):
                ms_id = workflow[idx]
                source_device_id = self.device_data["data_sources"][idx]
                for j in range(O):
                    for k in range(D):
                        dev_name = self.device_data["device_models"][k]
                        workflow_latencies.append(x[ms_id, j, k] * speed_lookup_table[self.operator_data["operator_ids"][j]][dev_name])
                        accuracies.append(x[ms_id, j, k]*self.operator_list[self.operator_data["operator_ids"][j]]["accuracy"])
                if idx == 0:
                    for j in range(O):
                        for k in range(D):
                            workflow_latencies.append(x[ms_id, j, k]*self.device_data["transmission_speed"][source_device_id][k])
                else:
                    for j1 in range(O):
                        for k1 in range(D):
                            for j2 in range(O):
                                for k2 in range(D):
                                    workflow_latencies.append(x[ms_id, j1, k1]*self.device_data["transmission_speed"][k1][k2]*x[1-workflow[idx-1],j1,k1])
            # objectives.append(accuracy/len(workflow) - max(0, 1-delay_tol/workflow_latency))
            objectives.append(sum(accuracies)/len(workflow) - max(0,  1-delay_tol / sum(workflow_latencies)))

        solver.Maximize(solver.Sum(objectives))

        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL:
            print(f"The maximized utility sum = {solver.Objective().Value()}\n")
            # Print the values of x_ij
            print("Values of decision variable Y:")
            # for i in range(T):
            #     for j in range(O):
            #         if y[i, j].solution_value() != 0:
            #             source_device_id = self.device_data["data_sources"][i]
            #             # print(f"y_{i}_{j} =", y[i, j].solution_value())
            #             print(f"device {source_device_id}(as data source) transmits to operator {j}")
            #
            # # Print the values of y_jk
            # print("Values of decision variable X:")
            # for j in range(O):
            #     for k in range(D):
            #         if x[j, k].solution_value() != 0:
            #             print(f"operator {j} is deployed on device {k}")
            #         # print(f"x_{j}_{k} =", x[j, k].solution_value())

            data_flow_count = 0

            print("Values of z_ijk:")
            solution = [None]*M
            best_utility = solver.Objective().Value()
            # for i in range(M):
            #     for j in range(O):
            #         for k in range(D):
            #             if x[i, j, k].solution_value() != 0:
            #                 op_id = self.operator_data["operator_ids"][j]
            #                 solution[i] = (op_id, k)
            #                 # print(f"data flow {data_flow_count}: ")
            #                 # source_device_id = self.device_data["data_sources"][i]
            #                 # print(f"sensor on device {source_device_id} transmits data to operator {j} deployed on device {k}")
            #                 # accuracy = self.operator_data["operator_accuracies"][j]
            #                 # delay = self.device_data["transmission_speed"][source_device_id][k] + self.operator_data["processing_speed"][j][k]
            #                 # print(f"accuracy: {accuracy}")
            #                 # print(f"delay: {delay}")
            #                 #
            #                 # print("------------------------------------------------------------")
            #                 # data_flow_count += 1
            return solution, best_utility

        else:
            print("No optimal solution found.")
            return None



