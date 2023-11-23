import numpy as np
from ortools.linear_solver import pywraplp


speed_lookup_table = {
  1: {
    "jetson-nano": 4.364,
    "raspberrypi-4b": 7.0823,
    "jetson-xavier": 2.6235
  },
  0: {
    "jetson-nano": 0.5549,
    "raspberrypi-4b": 1.0702,
    "jetson-xavier": 0.4276
  },
  3: {
    "jetson-nano": 4.4829,
    "raspberrypi-4b": 7.2191,
    "jetson-xavier": 3.8648
  },
  2: {
    "jetson-nano": 0.5864,
    "raspberrypi-4b": 1.0913,
    "jetson-xavier": 0.4605
  }
}

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
            # device_names = [dev["model"] for dev in devices]
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

    def inverse(self, x):
        if x == 0:
            return 0
        else:
            return 1 / x

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


        # Each operator is deployed to at most 1 device.
        for j in range(O):
            solver.Add(solver.Sum([x[i, j, k] for i in range(T) for k in range(D)]) <= 1)

        # Each data source transmit to only one operator
        for i in range(T):
            solver.Add(solver.Sum([x[i,j,k] for j in range(O) for k in range(D)]) == 1)

        # the operator type should be consistent with tasks req
        for i in range(T):
            for j in range(O):
                for k in range(D):
                    solver.Add(x[i, j, k]*self.tasks[i]["object_code"] == x[i, j, k]*self.operator_data["operator_types"][j])

        # operator requirement sum in each device should not exceed its capacity
        for k in range(D):
            for t in range(4):
                solver.Add(solver.Sum([x[i, j, k] * self.operator_data["resource_requirements"][j][t] for j in range(O) for i in range(T)]) <=
                       self.device_data["resource_capability"][k][t])

        # operator rate sum should not exceed operator processing rate
        for j in range(O):
            # find all sources (i) that transmit to j
            sum_j = []
            processing_rate = 0
            for i in range(T):
                for k in range(D):
                    rate = self.tasks[i]["rate"]
                    dev_name = self.device_data["device_models"][k]
                    op_id = self.operator_data["operator_ids"][j]
                    processing_rate += x[i,j,k] * (1 / speed_lookup_table[op_id][dev_name])
                    sum_j.append(x[i,j,k] * rate)
            solver.Add(solver.Sum(sum_j) <= processing_rate)
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
            # print(f"The maximized utility sum = {solver.Objective().Value()}\n")
            # # Print the values of x_ij
            # print("Values of decision variable Y:")
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
            #
            # data_flow_count = 0
            #
            # print("Values of z_ijk:")
            solution = [None]*T
            best_utility = solver.Objective().Value()
            for i in range(T):
                for j in range(O):
                    for k in range(D):
                        if x[i, j, k].solution_value() != 0:
                            op_id = self.operator_data["operator_ids"][j]
                            solution[i] = (op_id, k)
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



