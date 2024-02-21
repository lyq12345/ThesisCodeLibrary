import copy
import json
import numpy as np
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

    def calculate_bid(self, mapping, microservice):
        pass

    def device_fail(self):
        pass
    def device_disconnection(self):
        pass
    def emergent_request(self):
        pass
    def auction_based_recovery(self, devices, unmicroservices, current_solution):
        pass


if __name__ == '__main__':
    adaptation = Adaptation()
    print("acc")