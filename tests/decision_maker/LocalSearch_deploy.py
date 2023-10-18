import numpy as np
import math


speed_lookup_table = {
  "joelee0515/firedetection:yolov3-measure-time": {
    "jetson-nano": 4.364,
    "raspberrypi-4b": 7.0823,
    "jetson-xavier": 2.6235
  },
  "joelee0515/firedetection:tinyyolov3-measure-time": {
    "jetson-nano": 0.5549,
    "raspberrypi-4b": 1.0702,
    "jetson-xavier": 0.4276
  },
  "joelee0515/humandetection:yolov3-measure-time": {
    "jetson-nano": 4.4829,
    "raspberrypi-4b": 7.2191,
    "jetson-xavier": 3.8648
  },
  "joelee0515/humandetection:tinyyolov3-measure-time": {
    "jetson-nano": 0.5864,
    "raspberrypi-4b": 1.0913,
    "jetson-xavier": 0.4605
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
    def __init__(self, tasks, devices, operators):
        self.tasks = tasks
        self.devices = devices
        self.operators = operators
        self.transmission_matrix = self.generate_transmission_rate_matrix(len(devices))


    def generate_transmission_rate_matrix(self, n, min_rate=5, max_rate=15):
        transmission_matrix = np.full((n, n), np.inf)

        # 对角线上的元素设为正无穷
        np.fill_diagonal(transmission_matrix, 0)

        # 随机生成不同device之间的传输速率并保持对称性
        for i in range(n):
            for j in range(i + 1, n):
                rate = np.random.randint(min_rate, max_rate + 1)  # 生成随机速率
                transmission_matrix[i, j] = rate
                transmission_matrix[j, i] = rate  # 对称性

        return transmission_matrix

    def is_system_consistent(self, system_resources, system_requirements):
        for key, value in system_requirements.items():
            if key not in system_resources:
                return False
            if key in system_resources:
                if isinstance(value, int) or isinstance(value, float):
                    if system_resources[key] < system_resources[key]:
                        return False
                else:
                    if system_requirements[key] != system_resources[key]:
                        return False

        return True

    def filter_devices(self, operator):
        filtered_devices = []
        for dev in self.devices:
            if self.is_system_consistent(dev["resources"]["system"], operator["requirements"]["system"]):
                filtered_devices.append(dev)
        return filtered_devices

    def calculate_delay(self, operator, source_device_id, device_id):
        operator_name = operator["name"]
        device_model = self.devices[device_id]["model"]
        transmission_delay = self.transmission_matrix[source_device_id, device_id]
        processing_delay = speed_lookup_table[operator_name][device_model]
        return transmission_delay + processing_delay

    def calculate_power(self, operator, device_id):
        operator_name = operator["name"]
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
        for operator in operator_candidates:
            filtered_devices = self.filter_devices(operator)
            for i in range(len(filtered_devices)):
                device_id = filtered_devices[i]["id"]
                accuracy = operator["accuracy"]
                delay = self.calculate_delay(operator, source_device_id, device_id)
                power = self.calculate_power(operator, device_id)
                criteria_list = [accuracy, delay, power]
                decision_matrix.append(criteria_list)
                mappings.append([operator["name"], device_id])

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


    def make_decision(self):
        print("Running TOPSIS decision maker")
        for task in self.tasks:
            object_code = task["object_code"]
            source_device_id = task["source"]
            operator_candidates = []
            for op in self.operators:
                if op["object_code"] == object_code:
                    operator_candidates.append(op)

            mapping, RC = self.calculate_rc(source_device_id, operator_candidates)
            print(mapping, RC)




