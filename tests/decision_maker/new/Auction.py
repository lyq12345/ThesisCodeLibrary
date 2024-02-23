import copy
import json
import numpy as np
import random
from pathlib import Path
import sys
import os
import time
sys.path.append(str(Path(__file__).resolve().parents[1]))
from tests.status_tracker.rescons_models import cpu_consumption

cur_dir = os.path.dirname(os.path.abspath(__file__))

speed_lookup_table = None
power_lookup_table = None
with open(os.path.join(cur_dir, "../../status_tracker/speed_lookup_table.json"), 'r') as file:
    speed_lookup_table = json.load(file)

with open(os.path.join(cur_dir, "../../status_tracker/power_lookup_table.json"), 'r') as file:
    power_lookup_table = json.load(file)
# 从 JSON 文件中读取数据
def read_json(filename):
    with open(filename, 'r') as json_file:
        data = json.load(json_file)
    return data

class Adaptation:
    def __init__(self):
        self.workflows = read_json("../mock/workflows.json")
        self.microservice_data = read_json("../mock/microservicedata.json")
        # """
        # microservices_data = {
        #     "microservices_graph": None,
        #     "ms_wf_mapping": None,
        #     "ms_types": None,
        # }
        # """
        self.ms_to_wf = []
        for wf_id, workflow in enumerate(self.workflows):
            microservices = workflow["workflow"]
            for _ in microservices:
                self.ms_to_wf.append(wf_id)
        self.ms_neigbbors = [[-1,-1] for _ in range(len(self.microservice_data["ms_types"]))]
        self.find_ms_neighbors()
        self.devices = read_json("../mock/devices.json")
        self.operator_data = read_json("../mock/operatordata.json")
        self.operator_profiles = read_json("../../status_tracker/operators.json")
        self.operator_loads = [0 for _ in range(len(self.operator_data))]

        self.transmission_matrix = np.load('../mock/transmission.npy')
        self.solution = read_json("../mock/solution.json")
        self.consume_operators()
        self.banned_devices = []

    def find_ms_neighbors(self):
        for ms_id in range(len(self.microservice_data["ms_types"])):
            for ms_id_2 in range(len(self.microservice_data["ms_types"])):
                if self.microservice_data["microservices_graph"][ms_id_2][ms_id] == 1:
                    self.ms_neigbbors[ms_id][0] = ms_id_2
                if self.microservice_data["microservices_graph"][ms_id][ms_id_2] == 1:
                    self.ms_neigbbors[ms_id][1] = ms_id_2

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
    def operator_reusable(self, mapping, rate):
        op_id = mapping[0]
        op_code = mapping[1]
        dev_id = mapping[2]
        dev_name = self.devices[dev_id]["model"]
        # find operator op_id's load
        operator_load = mapping[3]
        new_load = operator_load + rate
        if new_load > 1/speed_lookup_table[op_code][dev_name]:
            return False
        cpu_extra = cpu_consumption(op_code, dev_name, new_load)-cpu_consumption(op_code, dev_name, operator_load)
        if self.devices[dev_id]["resources"]["system"]["cpu"]<cpu_extra:
            return False

        return True

    def deploy(self, mapping):
        op_code = mapping[1]
        dev_id = mapping[2]
        dev_name = self.devices[dev_id]["model"]
        op_load = mapping[3]
        resource_requirements = self.operator_profiles[op_code]["requirements"]["system"]
        resource_requirements["cpu"] = cpu_consumption(op_code, dev_name, op_load)

        for type, amount in resource_requirements.items():
            self.devices[dev_id]["resources"]["system"][type] -= amount

    def reuse(self, mapping, rate):
        # op_id = mapping[0]
        op_code = mapping[1]
        dev_id = mapping[2]
        dev_name = self.devices[dev_id]["model"]
        previous_load = mapping[3]
        new_load = previous_load + rate
        extra_cpu = cpu_consumption(op_code, dev_name, new_load) - cpu_consumption(op_code, dev_name, previous_load)

        self.devices[dev_id]["resources"]["system"]["cpu"] -= extra_cpu
        # self.operator_loads[op_id] += rate

    def consume_operators(self):
        deployed_op_ids = []
        for mapping in self.solution:
            if mapping[0] in deployed_op_ids:
                continue
            self.deploy(mapping)
            deployed_op_ids.append(mapping[0])

    def calculate_bid_mapping(self, mapping, service_code, rate, current_acc, current_delay, delay_tol, previous_dev_id, next_dev_id):
        """
        bidding rule: mappings bid on microservices;
        bidding rule: the increment for workflow objective for each microservice
        """
        op_code = mapping[1]
        if mapping != service_code:
            return 0
        if not self.operator_reusable(mapping, rate):
            return 0
        dev_id = mapping[2]
        dev_model = self.devices[dev_id]["model"]
        acc = self.operator_profiles[op_code]["accuracy"]
        latency = speed_lookup_table[op_code][dev_model]
        if not (previous_dev_id == -1 and next_dev_id == -1):
            latency += 0
        elif previous_dev_id != -1 and next_dev_id == -1:
            latency += self.transmission_matrix[previous_dev_id][dev_id]
        elif previous_dev_id == -1 and next_dev_id != -1:
            latency += self.transmission_matrix[dev_id][next_dev_id]
        else:
            latency += (self.transmission_matrix[previous_dev_id][dev_id] + self.transmission_matrix[dev_id][
                next_dev_id])
        new_acc = current_acc*acc
        new_delay = current_delay + latency
        current_objective = 0.5*current_acc - 0.5*max(0, (current_delay-delay_tol)/current_delay)
        new_objective = 0.5*new_acc - 0.5*max(0, (new_delay-delay_tol)/new_delay)
        return new_objective-current_objective

    def calculate_bid_device(self, dev_id, service_code, rate, current_acc, current_delay, delay_tol, previous_dev_id, next_dev_id):
        op_candidates = []
        dev_model = self.devices[dev_id]["model"]
        for op in self.operator_profiles:
            if op["object_code"] == service_code:
                op_candidates.append(op["id"])
        useful_op_candidates = []
        for op_code in op_candidates:
            op_resource = self.operator_profiles["requirements"]["system"]
            op_resource["cpu"] = cpu_consumption(op_code, dev_model, rate)
            if self.is_system_consistent(op_resource, self.devices[dev_id]["resources"]["system"]):
                useful_op_candidates.append(op_code)

        highest_bid = 0
        best_op_code = -1
        for op_code in useful_op_candidates:
            acc = self.operator_profiles[op_code]["accuracy"]
            latency = speed_lookup_table[op_code][dev_model]
            if not (previous_dev_id == -1 and next_dev_id == -1):
                latency += 0
            elif previous_dev_id != -1 and next_dev_id == -1:
                latency += self.transmission_matrix[previous_dev_id][dev_id]
            elif previous_dev_id == -1 and next_dev_id != -1:
                latency += self.transmission_matrix[dev_id][next_dev_id]
            else:
                latency += (self.transmission_matrix[previous_dev_id][dev_id] + self.transmission_matrix[dev_id][
                    next_dev_id])
            new_acc = current_acc * acc
            new_delay = current_delay + latency
            current_objective = 0.5 * current_acc - 0.5 * max(0, (current_delay - delay_tol) / current_delay)
            new_objective = 0.5 * new_acc - 0.5 * max(0, (new_delay - delay_tol) / new_delay)
            bid = new_objective - current_objective
            if bid > highest_bid:
                highest_bid = bid
                best_op_code = op_code
        return highest_bid, [0, best_op_code, dev_id, rate]

    def device_fail(self):
        active_devices = []
        banned_devices = []
        for mapping in self.solution:
            if mapping[2] not in active_devices:
                active_devices.append(mapping[2])
        crushed_dev_id = random.choice(active_devices)
        banned_devices.append(crushed_dev_id)
        missed_ms_ids = []
        for ms_id, mapping in enumerate(self.solution):
            if mapping[2] == crushed_dev_id:
                missed_ms_ids.append(ms_id)
                self.solution[ms_id] = []
        return missed_ms_ids, banned_devices

    def power_shortage(self):
        pass
    def device_disconnection(self):
        pass
    def emergent_request(self):
        pass
    def auction_based_recovery(self):
        workflow_accuracies = [1 for _ in range(len(self.workflows))]
        workflow_delays = [0 for _ in range(len(self.workflows))]
        # randomly generating faults
        missed_mids, banned_devices = self.device_fail()
        missed_ms_codes = [self.microservice_data["ms_types"][ms_id] for ms_id in missed_mids]
        missed_ms_rates = [0 for _ in range(len(missed_mids))]
        microservice_dev_neighbors = {key: [-1, -1] for key in missed_mids}
        # for id in missed_mids:
        #     microservice_neighbors[id]["previous"] = -1
        ms_id_global = 0
        for wf_id, workflow in enumerate(self.workflows):
            microservices = workflow["workflow"]
            source_dev_id = workflow["source"]
            for id in range(len(microservices)):
                if ms_id_global not in missed_mids:
                    mapping = self.solution[ms_id_global]
                    dev_id = mapping[2]
                    dev_name = self.devices[mapping[1]]["model"]
                    op_code = mapping[1]
                    acc = self.operator_profiles[op_code]["accuracy"]
                    op_delay = speed_lookup_table[op_code][dev_name]
                    workflow_accuracies[wf_id] *= acc
                    workflow_delays[wf_id] += op_delay
                    pre_ms_id = self.ms_neigbbors[ms_id_global][0]
                    next_ms_id = self.ms_neigbbors[ms_id_global][1]
                    if pre_ms_id == -1: # if the first microservice:
                        if source_dev_id not in banned_devices: # check if source device crashsed
                            microservice_dev_neighbors[ms_id_global][0] = source_dev_id
                    else: # not the first microservice
                        if pre_ms_id not in missed_mids:
                            pre_dev_id = self.solution[pre_ms_id][2]
                            microservice_dev_neighbors[ms_id_global][0] = pre_dev_id
                    if next_ms_id != -1:
                        if next_ms_id not in missed_mids:
                            next_dev_id = self.solution[next_ms_id][2]
                            microservice_dev_neighbors[ms_id_global][1] = next_dev_id

                    if microservice_dev_neighbors[ms_id_global][0] != -1:
                        pre_dev_id = microservice_dev_neighbors[ms_id_global][0]
                        workflow_delays[wf_id] += self.transmission_matrix[pre_dev_id][dev_id]
                    # if id == 0: # the first microservice
                    #     if source_dev_id not in missed_mids:
                    #         workflow_delays[wf_id] += self.transmission_matrix[source_dev_id][dev_id]
                    # else:
                    #     pre_dev_id = self.solution[ms_id_global-1][2]
                    #     if pre_dev_id not in missed_mids:
                    #         workflow_delays[wf_id] += self.transmission_matrix[pre_dev_id][dev_id]
                ms_id_global += 1

        start_time = time.time()
        while len(missed_mids) > 0:
            bidders_existing = []
            bidders_devices = []
            # service_code = self.microservice_data["ms_types"][ms_id]
            for mapping in self.solution:
                if len(mapping) > 0:
                    if self.operator_profiles[mapping[1]]["object_code"] in missed_ms_codes:
                        # if self.operator_reusable(mapping, rate):
                        bidders_existing.append(mapping)
            for dev_id, dev in enumerate(self.devices):
                if dev_id not in banned_devices:
                    bidders_devices.append(dev_id)
            highest_price = float("-inf")
            best_bidder = None
            bidded_ms_id = -1
            existing = True
            for ms_id in missed_mids:
                wf_id = self.ms_to_wf[ms_id]
                service_code = self.microservice_data["ms_types"][ms_id]
                # bidder_id = 0
                for mapping in bidders_existing:
                    price = self.calculate_bid_mapping(mapping, service_code, missed_ms_rates[ms_id],
                                                       workflow_accuracies[wf_id], workflow_delays[wf_id], self.workflows[wf_id]["delay"],
                                                       microservice_dev_neighbors[ms_id][0], microservice_dev_neighbors[ms_id][1])
                    if price > highest_price:
                        highest_price = price
                        best_bidder = mapping
                        bidded_ms_id = ms_id
                        existing = True
                for dev_id in bidders_devices:
                    price, mapping = self.calculate_bid_device(dev_id, service_code, missed_ms_rates[ms_id],
                                                       workflow_accuracies[wf_id], workflow_delays[wf_id], self.workflows[wf_id]["delay"],
                                                       microservice_dev_neighbors[ms_id][0], microservice_dev_neighbors[ms_id][1])
                    if price > highest_price:
                        highest_price = price
                        best_bidder = mapping
                        bidded_ms_id = ms_id
                        existing = False
            # change device states
            bidded_rate = missed_ms_codes[bidded_ms_id]
            new_load = best_bidder[3] + bidded_rate
            if existing:
                self.reuse(best_bidder, bidded_rate)
                self.solution[bidded_ms_id] = best_bidder
                for mapping in self.solution:
                    if mapping[0] == best_bidder[0]:
                        mapping[3] = new_load
            else:
                self.deploy(best_bidder)
                self.solution[bidded_ms_id] = best_bidder
            missed_ms_codes.remove(bidded_ms_id)
            bidded_service_code = self.microservice_data["ms_types"][bidded_ms_id]
            missed_ms_codes.remove(bidded_service_code)

            # workflow_accuracies
            workflow_accuracies *= self.operator_profiles[best_bidder[1]]["accuracy"]
            bidded_wf_id = self.ms_to_wf[bidded_ms_id]
            # workflow_delays
            if microservice_dev_neighbors[bidded_ms_id][0] != -1:
                pre_dev_id = microservice_dev_neighbors[bidded_ms_id][0]
                workflow_delays[bidded_wf_id] += self.transmission_matrix[pre_dev_id][best_bidder[2]]
            if microservice_dev_neighbors[bidded_ms_id][1] != -1:
                next_dev_id = microservice_dev_neighbors[bidded_ms_id][1]
                workflow_delays[bidded_wf_id] += self.transmission_matrix[best_bidder[2]][next_dev_id]
            # neighbors
            pre_ms_id = self.ms_neigbbors[bidded_ms_id]
            next_ms_id = self.ms_neigbbors[bidded_ms_id]
            if pre_ms_id == -1:
                if self.workflows[bidded_wf_id]["source"] not in banned_devices:
                    microservice_dev_neighbors[bidded_ms_id][0] = self.workflows[bidded_wf_id]["source"]
            else:
                if pre_ms_id not in missed_mids:
                    microservice_dev_neighbors[bidded_ms_id][0] = self.solution[pre_ms_id][2]
            if next_ms_id != -1:
                if next_ms_id not in missed_mids:
                    microservice_dev_neighbors[bidded_ms_id][1] = self.solution[next_ms_id][2]


        end_time = time.time()
        elasped_time = end_time - start_time









if __name__ == '__main__':
    adaptation = Adaptation()
    adaptation.auction_based_recovery()
