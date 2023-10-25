import numpy as np
from ortools.linear_solver import pywraplp


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

class MIP_Decider:
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

        for device in devices:
            data["resource_capability"].append([device["resources"]["system"][key] for key in device["resources"]["system"]])

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
            device_names = [dev["model"] for dev in devices]
            for op in operators:
                if op["type"] == "processing" and op["object"] == object_type:
                    op_name = op["name"]
                    data["operator_accuracies"].append(op["accuracy"])
                    data["resource_requirements"].append([op["requirements"]["system"][key] for key in op["requirements"]["system"]])
                    data["operator_types"].append(op["object_code"])
                    data["operator_ids"].append(op["id"])
                    data["processing_speed"].append([speed_lookup_table[op_name][dev_name] for dev_name in device_names])
                    data["power_consumptions"].append([speed_lookup_table[op_name][dev_name] for dev_name in device_names])
                    count += 1
        self.num_operators = count

        return data

    def generate_transmission_rate_matrix(self, n, min_rate=5, max_rate=15):
        # 创建一个n*n的矩阵，初始值设为正无穷
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

    def inverse(self, x):
        if x == 0:
            return 0
        else:
            return 1 / x

    def make_decision(self):
        print("Running MIP decision maker")
        # Create the MIP solver
        solver = pywraplp.Solver.CreateSolver('SCIP')

        T = self.num_tasks
        O = self.num_operators
        D = self.num_devices

        # Create binary decision variables x_ij for the binary matrix x
        x = {}  # x_jk - operator j on device k
        for j in range(O):
            for k in range(D):
                x[j, k] = solver.IntVar(0, 1, f'x_{j}_{k}')

        # Create binary decision variables y_jk for the binary vector y
        y = {}  # y_ij - data source i to operator j
        for i in range(T):
            for j in range(O):
                y[i, j] = solver.IntVar(0, 1, f'y_{i}_{j}')

        z = {}
        for i in range(T):
            for j in range(O):
                for k in range(D):
                    z[i, j, k] = solver.IntVar(0, 1, f'z_{i}_{j}_{k}')

        # Add constraints to represent x_ij * y_jk as binary variables
        for i in range(T):
            for j in range(O):
                for k in range(D):
                    solver.Add(z[i, j, k] <= x[j, k])
                    solver.Add(z[i, j, k] <= y[i, j])
                    solver.Add(z[i, j, k] >= x[j, k] + y[i, j] - 1)

        # Each operator is assigned to at most 1 device.
        for j in range(O):
            solver.Add(solver.Sum([x[j, k] for k in range(D)]) <= 1)

        # Each data source transmit to at  most one operator
        for i in range(T):
            solver.Add(solver.Sum([y[i, j] for j in range(O)]) <= 1)

        # the operator type should be consistent with tasks req
        for i in range(T):
            for j in range(O):
                solver.Add(y[i, j]*self.tasks[i]["object_code"] == y[i, j]*self.operator_data["operator_types"][j])

        # operators in y should be consistent with operators in x
            for j in range(O):
                solver.Add(solver.Sum([y[i, j] for i in range(T)]) == solver.Sum([x[j, k] for k in range(D)]))



        # operator requirement sum in each device should not exceed its capacity
        for k in range(D):
            for t in range(4):
                solver.Add(solver.Sum([x[j, k] * self.operator_data["resource_requirements"][j][t] for j in range(O)]) <=
                       self.device_data["resource_capability"][k][t])

        utilities = []
        for i in range(T):
            for j in range(O):
                for k in range(D):
                    source_device_id = self.device_data["data_sources"][i]
                    utilities.append(z[i, j, k] * self.operator_data["operator_accuracies"][j]
                                     - z[i, j, k] * max((1 - 10 * self.inverse(
                        (self.device_data["transmission_speed"][source_device_id][k] + self.operator_data["processing_speed"][j][k]))), 0))

        # transmission_times = {}
        # for i in range(D):
        #     for j in range(O):
        #         for k in range(D):
        #             transmission_times[i, j, k] = z[i, j, k]*device_data["transmission_speed"][i][k]

        # for i in range(D):
        #     for j in range(O):
        #         for k in range(D):
        #             utilities.append(z[i, j, k]*accuracy[i, j, k]-z[i, j, k]*transmission_times[i, j, k])

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
                        if z[i, j, k].solution_value() != 0:
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



