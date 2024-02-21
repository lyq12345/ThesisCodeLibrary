import copy
import json
import numpy as np
import random
from pathlib import Path
import sys
import os
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
        self.devices = read_json("../mock/devices.json")
        self.operator_data = read_json("../mock/operatordata.json")
        self.operator_profiles = read_json("../../status_tracker/operators.json")
        self.operator_loads = [0 for _ in range(len(self.operator_data))]

        self.transmission_matrix = np.load('../mock/transmission.npy')
        self.solution = read_json("../mock/solution.json")
        self.consume_operators()
        self.banned_devices = []
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

    def device_disconnection(self):
        pass
    def emergent_request(self):
        pass
    def auction_based_recovery(self):
        missed_mids, banned_devices = self.device_fail()
        microservice_neighbors = {key: [-1, -1] for key in missed_mids }
        # for id in missed_mids:
        #     microservice_neighbors[id]["previous"] = -1
        # TODO: each ms_id with its previous and next dev_ids
        ms_id_global = 0
        for wf_id, workflow in enumerate(self.workflows):
            microservices = workflow["workflow"]
            for id in range(len(microservices)):
                if ms_id_global in missed_mids:
                    if id==0:
                        microservice_neighbors[ms_id_global][0] = workflow["source"]
                    if id == len(microservices)-1:
                        microservice_neighbors[ms_id_global][1] = -1
                    if id != 0 and id != len(microservices)-1:
                        if len(self.solution[ms_id_global-1]) == 0:
                            microservice_neighbors[ms_id_global][0] = -1
                        else:
                            microservice_neighbors[ms_id_global][0] = self.solution[ms_id_global-1][2]
                        if len(self.solution[ms_id_global+1]) == 0:
                            microservice_neighbors[ms_id_global][1] = -1
                        else:
                            microservice_neighbors[ms_id_global][1] = self.solution[ms_id_global+1][2]
                ms_id_global += 1

        while len(missed_mids) > 0:
            bidders_existing = []
            bidders_devices = []
            for ms_id in missed_mids:
                service_code = self.microservice_data["ms_types"][ms_id]
                wf_id = self.ms_to_wf[ms_id]
                rate = self.workflows[wf_id]["rate"]
                for mapping in self.solution:
                    if self.operator_profiles[mapping[1]]["object_code"] == service_code:
                        if self.operator_reusable(mapping, rate):
                            bidders.append(mapping)





if __name__ == '__main__':
    adaptation = Adaptation()
    adaptation.auction_based_recovery()