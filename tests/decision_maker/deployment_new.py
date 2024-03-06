import json
import time
import os
import numpy as np
import sys
import argparse
import pandas as pd
import random
from pathlib import Path
from enum import Enum

sys.path.append(str(Path(__file__).resolve().parents[1]))
# from greedy_deploy import Greedy_Decider
# from MIP_deploy import MIP_Decider
from new.utils import calculate_effective_transmission_time
from TOPSIS_deploy import TOPSIS_decider
from LocalSearch_deploy import LocalSearch_deploy
# from ORTools_deploy import ORTools_Decider
from new.ORTools_deploy import ORTools_Decider
from new.Greedy_deploy import Greedy_decider
from new.LocalSearch_deploy import LocalSearch_new
from new.ILS import Iterated_LS_decider
from new.ODP_LS import ODP_LS_Decider
from new.ODP_TabuSearch import ODP_TS_Decider
from new.SA_deploy import SA_Decider
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from status_tracker.rescons_models import cpu_consumption
# from status_tracker.task_mock import generate_tasks
from status_tracker.workflow_mock import generate_workflows
from status_tracker.device_mock import generate_devices
from status_tracker.AP_mock import generate_access_points

cur_dir = os.getcwd()

speed_lookup_table = None
power_lookup_table = None
with open(os.path.join(cur_dir, "../status_tracker/speed_lookup_table.json"), 'r') as file:
    speed_lookup_table = json.load(file)
with open(os.path.join(cur_dir, "../status_tracker/power_lookup_table.json"), 'r') as file:
    power_lookup_table = json.load(file)

data = {'group': [], 'Normalized objective': [], 'time': [], 'algorithm': [], 'CPU usage': [], 'Memory usage': []}

def read_json(filename):
    with open(filename, 'r') as json_file:
        data = json.load(json_file)
    return data

def print_table(data):
    # 打印表头
    print("|", end="")
    for header in data[0]:
        print(f" {header} |", end="")
    print()

    # 打印分隔线
    print("|", end="")
    for _ in data[0]:
        print("------|", end="")
    print()

    # 打印数据行
    for row in data[1:]:
        print("|", end="")
        for item in row:
            print(f" {item} |", end="")
        print()

class MyEnum(Enum):
    WIFI = 1
    FOURG = 2
    LORA = 3
    BLUETOOTH = 4

def generate_network_model(n):
    probabilities = [0.8, 0.1, 0.05, 0.01]
    bw_dict = {
        "BLUETOOTH": [2],
        "WIFI": [11],
        "FOURG": [20],
        "LORA": [0.01]
    }
    delay_dict = {
        "BLUETOOTH": [1, 100],
        "WIFI": [1, 100],
        "FOURG": [10, 100],
        "LORA": [100, 1000]
    }
    # delay_dict = [[1, 100], [10, 100], [1, 100], [100, 1000]]
    # bw_dict = [[11], [20], [2], [0.01]]

    delay_matrix = np.full((n, n), 0.0)
    bandwidth_matrix = np.full((n, n), np.inf)
    linktype_matrix = np.full((n, n), 0)
    for i in range(n):
        for j in range(i+1, n):
            if i != j:
                link_type = random.choices(list(MyEnum), weights=probabilities)[0]
                linktype_matrix[i][j] = link_type.value
                linktype_matrix[j][i] = link_type.value
                delay = random.uniform(delay_dict[link_type.name][0], delay_dict[link_type.name][1])
                bandwidth = bw_dict[link_type.name][0]
                delay_matrix[i][j] = delay
                delay_matrix[j][i] = delay
                bandwidth_matrix[i][j] = bandwidth
                bandwidth_matrix[j][i] = bandwidth

    return delay_matrix, bandwidth_matrix, linktype_matrix

def generate_transmission_rate_matrix(n, min_rate=1, max_rate=5):
    transmission_matrix = np.full((n, n), np.inf)

    # diagnose to zero
    np.fill_diagonal(transmission_matrix, 0)

    # get random link rates between pairs
    for i in range(n):
        for j in range(i + 1, n):
            rate = random.uniform(min_rate, max_rate)
            transmission_matrix[i, j] = rate
            transmission_matrix[j, i] = rate  # (i,j) = (j,i)

    return transmission_matrix

def create_microservice_model(workflows):
    ms_num = sum([len(item["workflow"]) for item in workflows])
    microservices_data = {
        "microservices_graph": None,
        "ms_wf_mapping": None,
        "ms_types": None,
    }
    microservices_data["microservices_graph"] = [[0 for _ in range(ms_num)] for _ in range(ms_num)]
    microservices_data["ms_wf_mapping"] = [[0 for _ in range(len(workflows))] for _ in range(ms_num)] #[ms_id][wf_id]
    microservices_data["ms_types"] = []
    ms_count = 0
    for wf_id, workflow in enumerate(workflows):
        microservices = workflow["workflow"]
        for i in range(len(microservices)):
            ms_type = microservices[i]
            if i > 0:
                microservices_data["microservices_graph"][ms_count-1][ms_count] = 1
            microservices_data["ms_types"].append(ms_type)
            microservices_data["ms_wf_mapping"][ms_count][wf_id] = 1
            ms_count += 1
    return microservices_data

def create_operator_model(operators, ms_types):
    operator_candidates = []
    for type in ms_types:
        for op_code, op in enumerate(operators):
            if op["object_code"] == type:
                operator_candidates.append(op_code)
    return operator_candidates



def make_decision_from_task_new(workflow_list, microservice_data, operator_data, device_list, transmission_matrix, solver="LocalSearch", display=True,
                                record=False, iterations=1):
    operator_file = os.path.join(cur_dir, "../status_tracker/operators.json")
    operator_list = read_json(operator_file)

    def calculate_accuracy(op_code):
        return operator_list[op_code]["accuracy"]

    def calculate_delay(operator_id, source_device_id, device_id):
        device_model = device_list[device_id]["model"]
        transmission_delay = transmission_matrix[source_device_id, device_id]
        processing_delay = speed_lookup_table[operator_id][device_model]
        return transmission_delay + processing_delay

    def calculate_objective(op_id, source_device_id, device_id, task_delay):
        acc = calculate_accuracy(op_id)
        delay = calculate_delay(op_id, source_device_id, dev_id)
        objective = acc - max(0, (delay - task_delay) / delay)
        return objective

    def calculate_power(operator_id, device_id):
        # operator_name = operator_list[operator_id]["name"]
        device_model = device_list[device_id]["model"]
        power = power_lookup_table[operator_id][device_model]
        return power

    def calculate_resource_consumption(solution):
        cpu_consumptions = [0] * len(device_list)
        ram_consumptions = [0] * len(device_list)
        cpu_sum = 0.0
        memory_sum = 0.0
        for dev in device_list:
            cpu_sum += dev["resources"]["system"]["cpu"]
            memory_sum += dev["resources"]["system"]["memory"]
        deployed_op_ids = []
        for i, mapping in enumerate(solution):
            op_id = mapping[0]
            if op_id in deployed_op_ids:
                continue
            op_code = mapping[1]
            dev_id = mapping[2]
            dev_name = device_list[dev_id]["model"]
            op_load = mapping[3]

            op_resource = operator_list[op_code]["requirements"]["system"]
            cpu_cons = cpu_consumption(op_code, dev_name, op_load)
            cpu_consumptions[dev_id] += cpu_cons
            ram_consumptions[dev_id] += op_resource["memory"]
            deployed_op_ids.append(op_id)

        avg_cpu_consumption = sum(cpu_consumptions) / len(cpu_consumptions)
        avg_ram_consumption = sum(ram_consumptions) / len(ram_consumptions)
        print(cpu_consumptions)
        cpu_percentage = sum(cpu_consumptions) / cpu_sum

        memory_percentage = sum(ram_consumptions) / memory_sum
        return cpu_percentage, memory_percentage

    decision_maker = None

    sum_elapsed_time = 0.0
    sum_utility = 0.0
    sum_cpu_usage = 0.0
    sum_memory_usage = 0.0

    res_objective = 0
    res_time = 0

    for i in range(iterations):
        # print(f"Running iteration {i + 1} ...")
        if solver == "TOPSIS":
            decision_maker = TOPSIS_decider(workflow_list, microservice_data, operator_data, device_list, operator_list,
                                            transmission_matrix)
        # elif solver == "LocalSearch":
        #     decision_maker = LocalSearch_deploy(workflow_list, microservice_data, operator_data, device_list,
        #                                         operator_list, transmission_matrix)
        # elif solver == "ORTools":
        #     decision_maker = ORTools_Decider(workflow_list, microservice_data, operator_data, device_list,
        #                                      operator_list, transmission_matrix)
        # elif solver == "Greedy":
        #     decision_maker = Greedy_decider(workflow_list, microservice_data, operator_data, device_list, operator_list, transmission_matrix)
        elif solver == "LocalSearch_new":
            decision_maker = LocalSearch_new(workflow_list, microservice_data, operator_data, device_list,
                                             operator_list, transmission_matrix)
        elif solver == "Greedy_accfirst":
            decision_maker = Greedy_decider(workflow_list, microservice_data, operator_data, device_list, operator_list,
                                            transmission_matrix, "accfirst")
        elif solver == "Greedy_delayfirst":
            decision_maker = Greedy_decider(workflow_list, microservice_data, operator_data, device_list, operator_list,
                                            transmission_matrix, "delayfirst")
        elif solver == "Greedy_multi":
            decision_maker = Greedy_decider(workflow_list, microservice_data, operator_data, device_list, operator_list,
                                            transmission_matrix, "multi")
        elif solver == "SA":
            decision_maker = SA_Decider(workflow_list, microservice_data, operator_data, device_list, operator_list,
                                            transmission_matrix)
        elif solver == "ILS":
            decision_maker = Iterated_LS_decider(workflow_list, microservice_data, operator_data, device_list, operator_list,
                                            transmission_matrix)
        elif solver == "ODP-LS":
            decision_maker = ODP_LS_Decider(workflow_list, microservice_data, operator_data, device_list, operator_list,
                                            transmission_matrix)
        elif solver == "ODP-TS":
            decision_maker = ODP_TS_Decider(workflow_list, microservice_data, operator_data, device_list, operator_list,
                                            transmission_matrix)
        start_time = time.time()
        solution, utility = decision_maker.make_decision()
        res_objective = utility
        end_time = time.time()
        elapsed_time = end_time - start_time
        res_time = elapsed_time

        cpu_usage, memory_usage = calculate_resource_consumption(solution)

        sum_elapsed_time += elapsed_time
        sum_utility += utility
        sum_cpu_usage += cpu_usage
        sum_memory_usage += memory_usage

        if display:
            print("Solution: ")
            ms_id = 0
            utility_sum = 0.0
            for wf_id, workflow in enumerate(workflow_list):
                microservices = workflow["workflow"]
                delay_tol = workflow["delay"]
                rate = workflow["rate"]
                source_device_id = workflow["source"]
                accuracy = 1.0
                workflow_latency = 0.0
                unsatisfied = False
                for i in range(len(microservices)):
                    mapping = solution[ms_id]
                    if len(mapping) == 0:
                        unsatisfied = True
                        break
                    op_code = mapping[1]
                    dev_id = mapping[2]
                    dev_name = device_list[dev_id]["model"]

                    operator_acc = operator_list[op_code]["accuracy"]
                    accuracy *= operator_acc
                    operator_latency = speed_lookup_table[op_code][dev_name]
                    workflow_latency += operator_latency

                    if i == 0:
                        workflow_latency += transmission_matrix[source_device_id][dev_id]
                    else:
                        previous_dev_id = solution[ms_id-1][2]
                        workflow_latency += transmission_matrix[previous_dev_id][dev_id]
                    # if performance_delay > task_delay:
                    #     delay_deviation_sum += round((performance_delay-task_delay)/task_delay, 2)
                    # performance_power = calculate_power(op_id, dev_id)
                    # performance_objective = calculate_objective(op_id, source_device_id, dev_id, task_delay)
                    ms_id += 1
                print(f"Workflow {wf_id}:")
                if unsatisfied:
                    print("unsatisfied!")
                    continue
                # utility = ((0.5*accuracy - 0.5*max(0, (workflow_latency-delay_tol)/workflow_latency))+1)/2
                # utility_sum += utility

                print(f"Request: source: {source_device_id} microservices: {microservices}, delay tolerance: {delay_tol}")
                # print(f"Deployment: {op_name} -> device {dev_id}")
                print(
                    f"Performance: accuracy: {accuracy}, latency: {workflow_latency}")
                print("--------------------------------------------------------------")
            print(f"Decision making time: {elapsed_time} s")
            print("Resource consumption: ")
            # cpu_usage, memory_usage = calculate_resource_consumption(solution)
            print(f"CPU Usage: {cpu_usage}")
            print(f"Memory Usage: {memory_usage}")
            print(f"Overall Objective: {res_objective}")

    avg_nol_objective = (sum_utility / iterations) / len(workflow_list)
    avg_time = sum_elapsed_time / iterations
    avg_cpu_usage = sum_cpu_usage / iterations
    avg_memory_usage = sum_memory_usage / iterations
    if record:
        data['Normalized objective'].append(avg_nol_objective)
        data['time'].append(avg_time)
        data['CPU usage'].append(avg_cpu_usage)
        data['Memory usage'].append(avg_memory_usage)
        data['group'].append(len(workflow_list))
        data['algorithm'].append(solver)
    return avg_nol_objective, avg_time, avg_cpu_usage, avg_memory_usage


def main():
    operator_file = os.path.join(cur_dir, "../status_tracker/operators.json")
    operator_list = read_json(operator_file)

    parser = argparse.ArgumentParser(description='test script.')

    parser.add_argument('-d', '--num_devices', default=30, type=int, help='number of devices')
    parser.add_argument('-r', '--num_requests', default=30, type=float, help='number of requests')
    parser.add_argument('-s', '--solver', type=str, default='All', help='solver name')
    parser.add_argument('-i', '--iterations', type=str, default=1, help='iteration times')

    args = parser.parse_args()

    num_devices = args.num_devices
    num_requests = args.num_requests
    solver = args.solver
    iterations = args.iterations

    table_data = [
        ["Algorithm", "Objective", "Time"]
    ]

    all_algorithms = ["Greedy_accfirst", "Greedy_delayfirst", "Greedy_multi", "LocalSearch_new", "ILS", "ODP-LS", "ODP-TS"]
    sum_times = []
    sum_objectives = []
    if solver == "All":
        for _ in all_algorithms:
            sum_times.append(0.0)
            sum_objectives.append(0.0)
    else:
        sum_times.append(0.0)
        sum_objectives.append(0.0)
    for _ in range(iterations):
        device_list = generate_devices(num_devices)
        workflow_list = generate_workflows(num_requests, device_list)
        microservice_data = create_microservice_model(workflow_list)
        operator_data = create_operator_model(operator_list, microservice_data["ms_types"])
        transmission_matrix = generate_transmission_rate_matrix(len(device_list))
        # delay_matrix, bandwidth_matrix, linktype_matrix = generate_network_model(len(device_list))
        access_points = generate_access_points(-20, 80, -20, 60, 0, 60, 5)
        calculate_effective_transmission_time(device_list, access_points)


        if solver == "All":
            for i in range(len(all_algorithms)):
                algorithm = all_algorithms[i]
                obj, time, cpu, memory = make_decision_from_task_new(workflow_list, microservice_data, operator_data, device_list, transmission_matrix, algorithm)
                sum_times[i] += time
                sum_objectives[i] += obj
        else:
            obj, time, cpu, memory = make_decision_from_task_new(workflow_list, microservice_data, operator_data, device_list,
                                                    transmission_matrix, solver)
            sum_times[0] += time
            sum_objectives[0] += obj
            # obj_1, time_1 = make_decision_from_task_new(workflow_list, microservice_data, operator_data, device_list, transmission_matrix, "Greedy_accfirst")
            # table_data.append(["Greedy_accfirst", obj_1, time_1])
            #
            # obj_2, time_2 = make_decision_from_task_new(workflow_list,microservice_data, operator_data, device_list, transmission_matrix, "Greedy_delayfirst")
            # table_data.append(["Greedy_delayfirst", obj_2, time_2])
            #
            # obj_3, time_3 = make_decision_from_task_new(workflow_list,microservice_data, operator_data, device_list, transmission_matrix, "Greedy_multi")
            # table_data.append(["Greedy_multi", obj_3, time_3])
            #
            # # obj_4, time_4 = make_decision_from_task_new(workflow_list,microservice_data, operator_data, device_list, transmission_matrix, "ORTools")
            # # table_data.append(["ORTools", obj_4, time_4])
            #
            # obj_4, time_4 = make_decision_from_task_new(workflow_list,microservice_data, operator_data, device_list, transmission_matrix, "LocalSearch_new")
            # table_data.append(["LocalSearch_new", obj_4, time_4])

        # else:
        #     obj, time = make_decision_from_task_new(workflow_list, microservice_data, operator_data, device_list, transmission_matrix, solver)
        #     table_data.append([solver, obj, time])

    print("Summary:")
    avg_times = [item/iterations for item in sum_times]
    avg_objectives = [item / iterations for item in sum_objectives]
    for i in range(len(avg_times)):
        time = avg_times[i]
        obj = avg_objectives[i]
        algorithm = solver if solver!="All" else all_algorithms[i]
        table_data.append([algorithm, obj, time])
    print_table(table_data)
    # make_decision_from_task_new(task_list, device_list, transmission_matrix, "TOPSIS")
    # make_decision_from_task_new(task_list, device_list, transmission_matrix, "MIP")


def evaluation_experiments():
    operator_file = os.path.join(cur_dir, "../status_tracker/operators.json")
    operator_list = read_json(operator_file)
    num_devices = [30]
    num_requests = [i for i in range(10, 31, 10)]
    iterations = 3
    # num_devices = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    # num_requests = [10]
    solvers = ["Greedy_accfirst", "Greedy_delayfirst", "Greedy_multi", "LocalSearch_new", "ILS", "ODP-LS", "ODP-TS"]

    for i, device_num in enumerate(num_devices):
        # for j in range(i + 1):
        for j in range(len(num_requests)):
            request_num = num_requests[j]
            sum_times = [0.0 for _ in range(len(solvers))]
            sum_objectives = [0.0 for _ in range(len(solvers))]
            sum_cpu_usages = [0.0 for _ in range(len(solvers))]
            sum_memory_usages = [0.0 for _ in range(len(solvers))]
            for itr in range(iterations):
                device_list = generate_devices(device_num)
                workflow_list = generate_workflows(request_num, device_list)
                microservice_data = create_microservice_model(workflow_list)
                operator_data = create_operator_model(operator_list, microservice_data["ms_types"])
                transmission_matrix = generate_transmission_rate_matrix(len(device_list))
                for i, solver in enumerate(solvers):
                    print(f"Running i={request_num} k={device_num}, solver={solver}, iteration={itr+1}/{iterations}")
                    obj, time, cpu, memory = make_decision_from_task_new(workflow_list, microservice_data, operator_data, device_list, transmission_matrix, solver, display=False,
                                                record=False, iterations=1)
                    sum_times[i] += time
                    sum_objectives[i] += obj
                    sum_cpu_usages[i] += cpu
                    sum_memory_usages[i] += memory
            avg_times = [item / iterations for item in sum_times]
            avg_objectives = [item / iterations for item in sum_objectives]
            avg_cpus = [item / iterations for item in sum_cpu_usages]
            avg_memorys = [item / iterations for item in sum_memory_usages]
            for i in range(len(avg_times)):
                time = avg_times[i]
                obj = avg_objectives[i]
                cpu = avg_cpus[i]
                memory = avg_memorys[i]
                algorithm = solvers[i]
                data['Normalized objective'].append(obj)
                data['time'].append(time)
                data['CPU usage'].append(cpu)
                data['Memory usage'].append(memory)
                data['group'].append(request_num)
                data['algorithm'].append(algorithm)
    # record finishes, save into csv
    df = pd.DataFrame(data)
    df.to_csv('results/evaluation_21.csv', index=False)


if __name__ == '__main__':
    main()
    # evaluation_experiments()

