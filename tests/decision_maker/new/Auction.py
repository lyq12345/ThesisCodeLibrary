import copy
import json
import numpy as np
import random
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from status_tracker.rescons_models import cpu_consumption
# 从 JSON 文件中读取数据
def read_json(filename):
    with open(filename, 'r') as json_file:
        data = json.load(json_file)
    return data

class Adaptation:
    def __init__(self):
        self.workflows = read_json("../mock/workflows.json")
        # self.microservice_data = microservice_data
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

    def calculate_bid(self, mapping, microservice):
        pass

    def device_fail(self):
        active_devices = []
        for mapping in self.solution:
            if mapping[2] not in active_devices:
                active_devices.append(mapping[2])
        crushed_dev_id = random.choice(active_devices)
        self.banned_devices.append(crushed_dev_id)
        missed_ms_ids = []
        for ms_id, mapping in enumerate(self.solution):
            if mapping[2] == crushed_dev_id:
                missed_ms_ids.append(ms_id)
        return missed_ms_ids

    def device_disconnection(self):
        pass
    def emergent_request(self):
        pass
    def auction_based_recovery(self, devices, unmicroservices, current_solution):
        pass


if __name__ == '__main__':
    adaptation = Adaptation()
