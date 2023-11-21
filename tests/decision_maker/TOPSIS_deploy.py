import numpy as np
import math

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

class TOPSIS_decider:
    def __init__(self, tasks, devices, operators, transmission_matrix):
        self.tasks = tasks
        self.devices = devices
        self.operators = operators
        self.transmission_matrix = transmission_matrix

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
        for task_id, mapping in enumerate(solution):
            source_device_id = self.tasks[task_id]["source"]
            operator_id = mapping[0]
            device_id = mapping[1]
            accuracy = self.operators[operator_id]["accuracy"]
            delay = self.calculate_delay(operator_id, source_device_id, device_id)
            task_del = self.tasks[task_id]["delay"]
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
        operator_name = self.operators[operator_id]["name"]
        device_model = self.devices[device_id]["model"]
        power = power_lookup_table[operator_name][device_model]
        return power

    def deploy(self, mapping):
        operator_name = mapping[0]
        device_id = mapping[1]
        operator_resource = {}
        for op in self.operators:
            if operator_name == op["name"]:
                operator_resource = op["requirements"]["system"]

        for type, amount in operator_resource.items():
            self.devices[device_id]["resources"]["system"][type] -= amount


    def calculate_rc(self, source_device_id, operator_candidates):
        # create decision matrix accuracy | delay | power
        decision_matrix = []
        mappings = []
        num_criterias = 3
        for op_id in operator_candidates:
            filtered_device_ids = self.filter_devices(op_id)
            for dev_id in filtered_device_ids:
                accuracy = self.operators[op_id]["accuracy"]
                delay = self.calculate_delay(op_id, source_device_id, dev_id)
                power = self.calculate_power(op_id, dev_id)
                criteria_list = [accuracy, delay, power]
                decision_matrix.append(criteria_list)
                mappings.append([op_id, dev_id])

        decision_matrix_np = np.array(decision_matrix)
        # Calculate the Normalized Decision Matrix (NDM)
        # for i in range(len(decision_matrix)):
        #     denominator = np.sqrt(np.sum(np.square(decision_matrix_np[i, :])))
        #     for j in range(num_criterias):
        #         decision_matrix_np[i, j] = decision_matrix_np[i, j] / denominator
        for j in range(num_criterias):
            denominator = np.sqrt(np.sum(np.square(decision_matrix_np[:, j])))
            for i in range(len(decision_matrix)):
                decision_matrix_np[i, j] = decision_matrix_np[i, j] / denominator
        # Calculate the Weighted Normalized Decision Matrix (WNDM)
        # weights = [1/num_criterias for _ in range(num_criterias)]
        weights = [0.5, 0.3, 0.2]
        # Calculate the Normalized Decision Matrix (NDM)
        for i in range(len(decision_matrix)):
            for j in range(num_criterias):
                decision_matrix_np[i, j] = decision_matrix_np[i, j] * weights[j]

        # Determine the best solution (A+) and the worst solution (A−)
        max_of_accuracy = np.max(decision_matrix_np[:, 0])
        min_of_accuracy = np.min(decision_matrix_np[:, 0])
        min_of_delay = np.min(decision_matrix_np[:, 1])
        max_of_delay = np.max(decision_matrix_np[:, 1])
        min_of_power = np.min(decision_matrix_np[:, 2])
        max_of_power = np.max(decision_matrix_np[:, 2])
        A_plus = np.array([max_of_accuracy, min_of_delay, min_of_power])
        A_minus = np.array([min_of_accuracy, max_of_delay, max_of_power])

        # Calculate the Separation Measures (SM).
        SM_plus = np.zeros(len(decision_matrix))
        SM_minus = np.zeros(len(decision_matrix))
        for i in range(len(decision_matrix)):
            sum_square_plus = 0
            sum_square_minus = 0
            for j in range(num_criterias):
                sum_square_plus += (decision_matrix_np[i, j]-A_plus[j])**2
                sum_square_minus += ((decision_matrix_np[i, j]-A_minus[j])**2)
            SM_plus[i] = math.sqrt(sum_square_plus)
            SM_minus[i] = math.sqrt(sum_square_minus)

        RC = np.zeros(len(decision_matrix))
        for i in range(len(decision_matrix)):
            RC[i] = SM_minus[i] / (SM_plus[i] + SM_minus[i])

        max_rc = max(RC)
        selected_mapping_id = -1
        for i in range(len(decision_matrix)):
            if RC[i] == max_rc:
                selected_mapping_id = i
                break

        if selected_mapping_id != -1:
            self.deploy(mappings[selected_mapping_id])

        return mappings[selected_mapping_id], max_rc


    def make_decision(self, display=True):
        if display:
            print("Running TOPSIS decision maker")
        solution = []
        for task in self.tasks:
            object_code = task["object_code"]
            source_device_id = task["source"]
            operator_candidates = []
            for op in self.operators:
                if op["object_code"] == object_code:
                    operator_candidates.append(op["id"])

            mapping, RC = self.calculate_rc(source_device_id, operator_candidates)
            solution.append(mapping)
        utility = self.calculate_utility(solution)
        return solution, utility




