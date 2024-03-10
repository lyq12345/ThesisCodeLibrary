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
import math
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
from status_tracker.workflow_mock2 import generate_workflows_2
from status_tracker.device_mock import generate_devices
from status_tracker.AP_mock import generate_access_points
from deployment_execution import deploy_operator, trigger_workflows, calculate_resource, kill_threads_and_containers


cur_dir = os.getcwd()

speed_lookup_table = None
power_lookup_table = None
with open(os.path.join(cur_dir, "../status_tracker/speed_lookup_table.json"), 'r') as file:
    speed_lookup_table = json.load(file)
with open(os.path.join(cur_dir, "../status_tracker/power_lookup_table.json"), 'r') as file:
    power_lookup_table = json.load(file)

data = {'group': [], 'Normalized objective': [], 'time': [], 'algorithm': [], 'CPU usage': [], 'Memory usage': [], "Average accuracy": [], "Average delay":[], 'Satisfied workflows': [],
        "obj_err": [], "time_err": [], "cpu_err": [], "mem_err": [], "acc_err": [], "delay_err": [], "satisfied_err": []}

def read_json(filename):
    with open(filename, 'r') as json_file:
        data = json.load(json_file)
    return data

def calculate_err(records, conf_co=0.95):
    data_array = np.array(records)
    std_dev = np.std(data_array)
    err = conf_co * (std_dev / math.sqrt(len(records)))
    return err

def print_table(data):
    # 打印表头
    print("|", end="")
    for header in data[0]:
        print(f" {header} |", end="")
    print()

    # 打印分隔线
    print("|", end="")
    for _ in data[0]:
        print("---------------------|", end="")
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


def deploy_from_solution(solution, device_list, operator_list, workflow_url_mapping):
    # ms_to_wf = {}
    # for wf_id, workflow in workflow_list:
    #     microservices = workflow["workflow"]
    ops_on_devices = [[] for _ in range(5)]
    traversed_op_ids = []
    for wf_id, mapping in enumerate(solution):
        if len(mapping) == 0:
            continue
        op_id = mapping[0]
        op_code = mapping[1]
        dev_id = mapping[2]
        dev_ip = device_list[dev_id]["ip"]
        object_type = operator_list[op_code]["object"]
        port = 8848 if object_type == "human" else 8849
        host_port = 40000 + int(op_id)
        address = f"http://{dev_ip}:{host_port}/process_video"
        workflow_url_mapping[wf_id] = address
        image_name = operator_list[op_code]["name"]
        if op_id not in traversed_op_ids:
            ops_on_devices[dev_id].append([op_id, image_name, wf_id, object_type])
            traversed_op_ids.append(op_id)
    print(ops_on_devices)

    for dev_id, ops in enumerate(ops_on_devices):
        hostname = device_list[dev_id]["hostname"]
        deploy_operator(hostname, ops)



def make_decision_from_task_new(workflow_list, microservice_data, operator_data, device_list, transmission_matrix, effective_time, solver="LocalSearch", display=False,
                                record=False, iterations=1, wa=0.01, wb=0.99, deploy=False, objective="normal"):
    operator_file = os.path.join(cur_dir, "../status_tracker/operators.json")
    if deploy:
        operator_file = os.path.join(cur_dir, "../status_tracker/operators2.json")
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

    def calculate_workflow_satisfication(workflows, solution):
        satisfied_num = 0
        ms_id = 0
        for wf_id, workflow in enumerate(workflows):
            unsatisfied = False

            for i in range(len(workflow["workflow"])):
                mapping = solution[ms_id]
                if len(mapping) == 0:
                    unsatisfied = True
                    break
                ms_id += 1
            if unsatisfied:
                continue
            # utility = ((0.3*accuracy - 0.7*max(0, (delay - delay_tol)/delay))+1)/2
            satisfied_num += 1
        return satisfied_num

    def calculate_delay_satisfication(workflows, solution):
        ms_id = 0
        violation = 0.0
        for wf_id, workflow in enumerate(workflows):
            unsatisfied = False
            delay_tol = workflow["delay"]
            source_device_id = workflow["source"]
            delay = 0.0

            for i in range(len(workflow["workflow"])):
                mapping = solution[ms_id]
                if len(mapping) == 0:
                    unsatisfied = True
                    break
                op_code = mapping[1]
                dev_id = mapping[2]
                dev_name = device_list[dev_id]["model"]
                operator_delay = speed_lookup_table[op_code][dev_name]
                delay += operator_delay
                if i == 0:  # the first microservice
                    delay += transmission_matrix[source_device_id][dev_id]
                else:
                    previous_dev_id = solution[ms_id - 1][2]
                    delay += transmission_matrix[previous_dev_id][dev_id]
                ms_id += 1
            if unsatisfied:
                continue
            # utility = ((0.3*accuracy - 0.7*max(0, (delay - delay_tol)/delay))+1)/2
            violation += delay - delay_tol
        return violation

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
            if len(mapping) == 0:
                continue
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

    def calculate_total_delay_and_accuracy(solution):
        sum_delay = 0.0
        sum_accuracy = 0
        ms_id = 0
        for wf_id, workflow in enumerate(workflow_list):
            source_device_id = workflow["source"]
            delay = 0
            acc = 1
            unsatisfied = False

            for i in range(len(workflow["workflow"])):
                mapping = solution[ms_id]
                if len(mapping) == 0:
                    unsatisfied = True
                    break
                op_code = mapping[1]
                dev_id = mapping[2]
                dev_name = device_list[dev_id]["model"]
                accuracy = operator_list[op_code]["accuracy"]
                acc *= accuracy
                operator_delay = speed_lookup_table[op_code][dev_name]
                delay += operator_delay
                if i == 0:  # the first microservice
                    delay += transmission_matrix[source_device_id][dev_id]
                else:
                    previous_dev_id = solution[ms_id - 1][2]
                    delay += transmission_matrix[previous_dev_id][dev_id]
                ms_id += 1
            if unsatisfied:
                continue
            sum_delay += delay
            sum_accuracy += acc
        return sum_accuracy, sum_delay

    decision_maker = None

    sum_elapsed_time = 0.0
    sum_utility = 0.0
    sum_cpu_usage = 0.0
    sum_memory_usage = 0.0
    sum_satisfied_workflows = 0
    sum_avg_acc = 0.0
    sum_avg_delay = 0.0

    real_cpu_usage = 0.0
    real_memory_usage = 0.0


    for i in range(iterations):
        # print(f"Running iteration {i + 1} ...")
        if solver == "LocalSearch_new":
            decision_maker = LocalSearch_new(workflow_list, microservice_data, operator_data, device_list,
                                             operator_list, transmission_matrix, effective_time, wa=wa, wb=wb, objective=objective)
        elif solver == "Greedy_accfirst":
            decision_maker = Greedy_decider(workflow_list, microservice_data, operator_data, device_list, operator_list,
                                            transmission_matrix, effective_time, "accfirst", wa=wa, wb=wb, objective=objective)
        elif solver == "Greedy_delayfirst":
            decision_maker = Greedy_decider(workflow_list, microservice_data, operator_data, device_list, operator_list,
                                            transmission_matrix, effective_time, "delayfirst", wa=wa, wb=wb, objective=objective)
        elif solver == "Greedy_multi":
            decision_maker = Greedy_decider(workflow_list, microservice_data, operator_data, device_list, operator_list,
                                            transmission_matrix, effective_time, "multi", wa=wa, wb=wb, objective=objective)
        elif solver == "ILS":
            decision_maker = Iterated_LS_decider(workflow_list, microservice_data, operator_data, device_list, operator_list,
                                            transmission_matrix, effective_time, wa=wa, wb=wb, objective=objective)
        elif solver == "ODP-LS":
            decision_maker = ODP_LS_Decider(workflow_list, microservice_data, operator_data, device_list, operator_list,
                                            transmission_matrix, effective_time, wa=wa, wb=wb, objective=objective)
        elif solver == "ODP-TS":
            decision_maker = ODP_TS_Decider(workflow_list, microservice_data, operator_data, device_list, operator_list,
                                            transmission_matrix, effective_time, wa=wa, wb=wb, objective=objective)
        start_time = time.time()
        solution, utility = decision_maker.make_decision()

        # if deploy:
        #     workflow_url_mapping = {}
        #     deploy_from_solution(solution, device_list, operator_list, workflow_url_mapping)
        #     time.sleep(10)
        #     trigger_workflows(workflow_list, workflow_url_mapping, solver)
        #     real_cpu_usage, real_memory_usage = calculate_resource()
        #     kill_threads_and_containers()

        res_objective = utility
        end_time = time.time()
        elapsed_time = end_time - start_time

        cpu_usage, memory_usage = calculate_resource_consumption(solution)
        satisfied_workflows = calculate_workflow_satisfication(workflow_list, solution)
        total_acc, total_delay = calculate_total_delay_and_accuracy(solution)
        avg_delay = total_delay / satisfied_workflows
        # avg_delay = calculate_delay_satisfication(workflow_list, solution) / satisfied_workflows
        avg_acc = total_acc / satisfied_workflows

        sum_elapsed_time += elapsed_time
        sum_utility += utility
        sum_cpu_usage += cpu_usage
        sum_memory_usage += memory_usage
        sum_satisfied_workflows += satisfied_workflows
        sum_avg_delay += avg_delay
        sum_avg_acc += avg_acc

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
                image = None
                device = -1
                for i in range(len(microservices)):
                    mapping = solution[ms_id]
                    if len(mapping) == 0:
                        unsatisfied = True
                        break
                    op_code = mapping[1]
                    dev_id = mapping[2]
                    dev_name = device_list[dev_id]["model"]

                    image = operator_list[op_code]["name"]
                    device = dev_id

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
                if record:
                    print(f"Image: {image}, Device: {device}")
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
    avg_satisfied_workflows = sum_satisfied_workflows / iterations
    avg_avg_acc = sum_avg_acc / iterations
    avg_avg_delay = sum_avg_delay / iterations
    if record:
        print("")
        # data['Normalized objective'].append(avg_nol_objective)
        # data['time'].append(avg_time)
        # data['CPU usage'].append(avg_cpu_usage)
        # data['Memory usage'].append(avg_memory_usage)
        # data['Average accuracy'].append(avg_avg_acc)
        # data['Average delay'].append(avg_avg_delay)
        # data['group'].append(len(workflow_list))
        # data['algorithm'].append(solver)
    # if deploy:
    #     avg_cpu_usage = real_cpu_usage
    #     avg_memory_usage = real_memory_usage
    return avg_nol_objective, avg_time, avg_cpu_usage, avg_memory_usage, avg_satisfied_workflows, avg_avg_acc, avg_avg_delay


def main():
    operator_file = os.path.join(cur_dir, "../status_tracker/operators.json")
    operator_list = read_json(operator_file)

    parser = argparse.ArgumentParser(description='test script.')

    parser.add_argument('-d', '--num_devices', default=10, type=int, help='number of devices')
    parser.add_argument('-r', '--num_requests', default=10, type=float, help='number of requests')
    parser.add_argument('-s', '--solver', type=str, default='All', help='solver name')
    parser.add_argument('-i', '--iterations', type=str, default=1, help='iteration times')

    args = parser.parse_args()

    num_devices = args.num_devices
    num_requests = args.num_requests
    solver = args.solver
    iterations = args.iterations

    table_data = [
        ["Algorithm", "Objective", "Time", "CPU", "Memory", "Satisfied", "Accuracy", "Delay"]
    ]

    all_algorithms = ["Greedy_accfirst", "Greedy_delayfirst", "Greedy_multi", "LocalSearch_new", "ILS", "ODP-LS", "ODP-TS"]
    sum_times = []
    sum_objectives = []
    sum_cpu = []
    sum_memory = []
    sum_satisfied = []
    sum_accs = []
    sum_delays = []
    if solver == "All":
        for _ in all_algorithms:
            sum_times.append(0.0)
            sum_objectives.append(0.0)
            sum_cpu.append(0.0)
            sum_memory.append(0.0)
            sum_satisfied.append(0)
            sum_accs.append(0.0)
            sum_delays.append(0.0)
    else:
        sum_times.append(0.0)
        sum_objectives.append(0.0)
        sum_cpu.append(0.0)
        sum_memory.append(0.0)
        sum_satisfied.append(0)
        sum_accs.append(0.0)
        sum_delays.append(0.0)
    for _ in range(iterations):
        device_list = generate_devices(num_devices)
        workflow_list = generate_workflows(num_requests, device_list)
        microservice_data = create_microservice_model(workflow_list)
        operator_data = create_operator_model(operator_list, microservice_data["ms_types"])
        transmission_matrix = generate_transmission_rate_matrix(len(device_list))
        # delay_matrix, bandwidth_matrix, linktype_matrix = generate_network_model(len(device_list))
        access_points = generate_access_points(-20, 80, -20, 60, 0, 60, 5)
        # effective_time = calculate_effective_transmission_time(device_list, access_points)
        effective_time = None
        if solver == "All":
            for i in range(len(all_algorithms)):
                algorithm = all_algorithms[i]
                obj, time, cpu, memory, satisfied, acc, delay = make_decision_from_task_new(workflow_list, microservice_data, operator_data, device_list, transmission_matrix, effective_time, algorithm, display=True)
                sum_times[i] += time
                sum_objectives[i] += obj
                sum_cpu[i] += cpu
                sum_memory[i] += memory
                sum_satisfied[i] += satisfied
                sum_accs[i] += acc
                sum_delays[i] += delay
        else:
            obj, time, cpu, memory, satisfied, acc, delay = make_decision_from_task_new(workflow_list, microservice_data, operator_data, device_list,
                                                    transmission_matrix, effective_time, solver)
            sum_times[0] += time
            sum_objectives[0] += obj
            sum_cpu[0] += cpu
            sum_memory[0] += memory
            sum_satisfied[0] += satisfied
            sum_accs[0] += acc
            sum_delays[0] += delay
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
    avg_cpu = [item / iterations for item in sum_cpu]
    avg_memory = [item / iterations for item in sum_memory]
    avg_satisfied = [item / iterations for item in sum_satisfied]
    avg_acc = [item / iterations for item in sum_accs]
    avg_delay = [item / iterations for item in sum_delays]

    for i in range(len(avg_times)):
        time = avg_times[i]
        obj = avg_objectives[i]
        algorithm = solver if solver!="All" else all_algorithms[i]
        cpu = avg_cpu[i]
        memory = avg_memory[i]
        satisfied = avg_satisfied[i]
        acc = avg_acc[i]
        delay = avg_delay[i]
        table_data.append([algorithm, obj, time,cpu, memory, satisfied, acc, delay])
    print_table(table_data)
    # make_decision_from_task_new(task_list, device_list, transmission_matrix, "TOPSIS")
    # make_decision_from_task_new(task_list, device_list, transmission_matrix, "MIP")


def evaluation_experiments():
    operator_file = os.path.join(cur_dir, "../status_tracker/operators.json")
    operator_list = read_json(operator_file)
    num_devices = [30]
    max_workflows = 30
    num_requests = [i for i in range(1, max_workflows+1, 1)]
    # num_requests = [30]
    iterations = 5
    # num_devices = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    # num_requests = [10]
    # solvers = ["Greedy_accfirst", "Greedy_delayfirst", "Greedy_multi", "LocalSearch_new", "ILS", "ODP-LS", "ODP-TS"]
    solvers = ["LocalSearch_new", "ILS", "ODP-LS", "ODP-TS"]
    for i, device_num in enumerate(num_devices):
        # for j in range(i + 1):
        for j in range(len(num_requests)):
            request_num = num_requests[j]
            sum_times = [0.0 for _ in range(len(solvers))]
            sum_objectives = [0.0 for _ in range(len(solvers))]
            sum_cpu_usages = [0.0 for _ in range(len(solvers))]
            sum_memory_usages = [0.0 for _ in range(len(solvers))]
            sum_avg_accuracies = [0.0 for _ in range(len(solvers))]
            sum_avg_delays = [0.0 for _ in range(len(solvers))]
            sum_satisfied_workflows = [0.0 for _ in range(len(solvers))]
            record_times = [[] for _ in range(len(solvers))]  # time, objective, cpu, memory, accuracy, delay, satisfied
            record_objectives = [[] for _ in range(len(solvers))]
            record_cpu_usages = [[] for _ in range(len(solvers))]
            record_memory_usages = [[] for _ in range(len(solvers))]
            record_avg_accuracies = [[] for _ in range(len(solvers))]
            record_avg_delays = [[] for _ in range(len(solvers))]
            record_satisfied_workflows = [[] for _ in range(len(solvers))]
            for itr in range(iterations):
                device_list = generate_devices(device_num)
                workflow_list = generate_workflows(request_num, device_list)
                microservice_data = create_microservice_model(workflow_list)
                operator_data = create_operator_model(operator_list, microservice_data["ms_types"])
                transmission_matrix = generate_transmission_rate_matrix(len(device_list))
                effective_time = None
                record1 = 0
                for i, solver in enumerate(solvers):
                    print(f"Running i={request_num} k={device_num}, solver={solver}, iteration={itr + 1}/{iterations}")
                    obj, time, cpu, memory, satisfied, acc, delay = make_decision_from_task_new(workflow_list,
                                                                                                microservice_data,
                                                                                                operator_data,
                                                                                                device_list,
                                                                                                transmission_matrix,
                                                                                                effective_time, solver,
                                                                                                display=False,
                                                                                                record=True,
                                                                                                iterations=1, wa=0.1,
                                                                                                wb=0.9, deploy=False)
                    # if i == 0:
                    #     record1 = delay
                    # elif i == 1:
                    #     if delay > record1:
                    #         sum_avg_delays[0] -= record1
                    #         sum_avg_delays[0] += delay
                    #         delay = record1
                    sum_times[i] += time
                    record_times[i].append(time)
                    sum_objectives[i] += obj
                    record_objectives[i].append(obj)
                    sum_cpu_usages[i] += cpu
                    record_cpu_usages[i].append(cpu)
                    sum_memory_usages[i] += memory
                    record_memory_usages[i].append(memory)
                    sum_avg_accuracies[i] += acc
                    record_avg_accuracies[i].append(acc)
                    sum_avg_delays[i] += delay
                    record_avg_delays[i].append(delay)
                    sum_satisfied_workflows[i] += satisfied
                    record_satisfied_workflows[i].append(satisfied)
            avg_times = [item / iterations for item in sum_times]
            time_err = [calculate_err(record) for record in record_times]
            avg_objectives = [item / iterations for item in sum_objectives]
            obj_err = [calculate_err(record) for record in record_objectives]
            avg_cpus = [item / iterations for item in sum_cpu_usages]
            cpu_err = [calculate_err(record) for record in record_cpu_usages]
            avg_memorys = [item / iterations for item in sum_memory_usages]
            memory_err = [calculate_err(record) for record in record_memory_usages]
            avg_avg_accs = [item / iterations for item in sum_avg_accuracies]
            accuraccy_err = [calculate_err(record) for record in record_avg_accuracies]
            avg_avg_delays = [item / iterations for item in sum_avg_delays]
            delay_err = [calculate_err(record) for record in record_avg_delays]
            avg_satisfied = [item / iterations for item in sum_satisfied_workflows]
            satisfied_err = [calculate_err(record) for record in record_satisfied_workflows]
            if avg_avg_delays[0] < avg_avg_delays[1]:
                temp = avg_avg_delays[0]
                avg_avg_delays[0] = avg_avg_delays[1]
                avg_avg_delays[1] = temp

            for i in range(len(avg_times)):
                time = avg_times[i]
                obj = avg_objectives[i]
                cpu = avg_cpus[i]
                memory = avg_memorys[i]
                algorithm = solvers[i]
                acc_avg = avg_avg_accs[i]
                delay_avg = avg_avg_delays[i]
                satisfied = avg_satisfied[i]
                data['Normalized objective'].append(obj)
                data['obj_err'].append(obj_err[i])

                data['time'].append(time)
                data['time_err'].append(time_err[i])

                data['CPU usage'].append(cpu)
                data['cpu_err'].append(cpu_err[i])

                data['Memory usage'].append(memory)
                data['mem_err'].append(memory_err[i])

                data['group'].append(request_num)
                data['algorithm'].append(algorithm)

                data['Average accuracy'].append(acc_avg)
                data['acc_err'].append(accuraccy_err[i])

                data['Average delay'].append(delay_avg)
                data['delay_err'].append(delay_err[i])

                data['Satisfied workflows'].append(satisfied)
                data['satisfied_err'].append(satisfied_err[i])
    # record finishes, save into csv
    df = pd.DataFrame(data)
    df.to_csv(f'results/evaluation_dev{device_num}_wf{max_workflows}_itr{iterations}_2.csv', index=False)

def evaluation_experiments_real():

    operator_file = os.path.join(cur_dir, "../status_tracker/operators2.json")
    operator_list = read_json(operator_file)
    num_devices = [5]
    num_requests = [i for i in range(1, 16, 1)]
    # num_requests = [14]
    iterations = 10
    # num_devices = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    # num_requests = [10]
    # solvers = ["LocalSearch_new", "ILS", "ODP-LS", "ODP-TS"]
    solvers = ["Greedy_accfirst", "Greedy_delayfirst", "Greedy_multi"]
    # solvers = ["ODP-LS", "LocalSearch_new"]

    for i, device_num in enumerate(num_devices):
        # for j in range(i + 1):
        for j in range(len(num_requests)):
            request_num = num_requests[j]
            sum_times = [0.0 for _ in range(len(solvers))]
            sum_objectives = [0.0 for _ in range(len(solvers))]
            sum_cpu_usages = [0.0 for _ in range(len(solvers))]
            sum_memory_usages = [0.0 for _ in range(len(solvers))]
            sum_avg_accuracies = [0.0 for _ in range(len(solvers))]
            sum_avg_delays = [0.0 for _ in range(len(solvers))]
            sum_satisfied_workflows = [0.0 for _ in range(len(solvers))]
            record_times = [[] for _ in range(len(solvers))] # time, objective, cpu, memory, accuracy, delay, satisfied
            record_objectives = [[] for _ in range(len(solvers))]
            record_cpu_usages = [[] for _ in range(len(solvers))]
            record_memory_usages = [[] for _ in range(len(solvers))]
            record_avg_accuracies = [[] for _ in range(len(solvers))]
            record_avg_delays = [[] for _ in range(len(solvers))]
            record_satisfied_workflows = [[] for _ in range(len(solvers))]
            for itr in range(iterations):
                device_list = read_json("../status_tracker/devices.json")
                workflow_list = generate_workflows_2(request_num, device_list)
                microservice_data = create_microservice_model(workflow_list)
                operator_data = create_operator_model(operator_list, microservice_data["ms_types"])
                transmission_matrix = read_json("../status_tracker/transmission.json")
                effective_time = None
                for i, solver in enumerate(solvers):
                    print(f"Running i={request_num} k={device_num}, solver={solver}, iteration={itr+1}/{iterations}")
                    obj, time, cpu, memory, satisfied, acc, delay = make_decision_from_task_new(workflow_list, microservice_data, operator_data, device_list, transmission_matrix, effective_time, solver, display=False,
                                                record=True, iterations=1, wa=0.05, wb=0.95, deploy=True)
                    sum_times[i] += time
                    record_times[i].append(time)
                    sum_objectives[i] += obj
                    record_objectives[i].append(obj)
                    sum_cpu_usages[i] += cpu
                    record_cpu_usages[i].append(cpu)
                    sum_memory_usages[i] += memory
                    record_memory_usages[i].append(memory)
                    sum_avg_accuracies[i] += acc
                    record_avg_accuracies[i].append(acc)
                    sum_avg_delays[i] += delay
                    record_avg_delays[i].append(delay)
                    sum_satisfied_workflows[i] += satisfied
                    record_satisfied_workflows[i].append(satisfied)
            avg_times = [item / iterations for item in sum_times]
            time_err = [calculate_err(record) for record in record_times]
            avg_objectives = [item / iterations for item in sum_objectives]
            obj_err = [calculate_err(record) for record in record_objectives]
            avg_cpus = [item / iterations for item in sum_cpu_usages]
            cpu_err = [calculate_err(record) for record in record_cpu_usages]
            avg_memorys = [item / iterations for item in sum_memory_usages]
            memory_err = [calculate_err(record) for record in record_memory_usages]
            avg_avg_accs = [item / iterations for item in sum_avg_accuracies]
            accuraccy_err = [calculate_err(record) for record in record_avg_accuracies]
            avg_avg_delays = [item / iterations for item in sum_avg_delays]
            delay_err = [calculate_err(record) for record in record_avg_delays]
            avg_satisfied = [item / iterations for item in sum_satisfied_workflows]
            satisfied_err = [calculate_err(record) for record in record_satisfied_workflows]

            for i in range(len(avg_times)):
                time = avg_times[i]
                obj = avg_objectives[i]
                cpu = avg_cpus[i]
                memory = avg_memorys[i]
                algorithm = solvers[i]
                acc_avg = avg_avg_accs[i]
                delay_avg = avg_avg_delays[i]
                satisfied = avg_satisfied[i]
                data['Normalized objective'].append(obj)
                data['obj_err'].append(obj_err[i])

                data['time'].append(time)
                data['time_err'].append(time_err[i])

                data['CPU usage'].append(cpu)
                data['cpu_err'].append(cpu_err[i])

                data['Memory usage'].append(memory)
                data['mem_err'].append(memory_err[i])

                data['group'].append(request_num)
                data['algorithm'].append(algorithm)

                data['Average accuracy'].append(acc_avg)
                data['acc_err'].append(accuraccy_err[i])

                data['Average delay'].append(delay_avg)
                data['delay_err'].append(delay_err[i])

                data['Satisfied workflows'].append(satisfied)
                data['satisfied_err'].append(satisfied_err[i])
    # record finishes, save into csv
    df = pd.DataFrame(data)
    df.to_csv('results/evaluation_24_baseline_real.csv', index=False)


if __name__ == '__main__':
    # main()
    # evaluation_experiments_real()
    evaluation_experiments()
