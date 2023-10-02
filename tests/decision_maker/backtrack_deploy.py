from typing import List, Dict
def check_constraints(operator: Dict, device: Dict) -> bool:
    op_constraint = operator['constraints']
    """检查应用部署条件是否满足"""
    for key, value in op_constraint.items():
        if key not in device and key not in device['resources']:
            return False
        if key in device:
            if isinstance(value, int) or isinstance(value, float):
                if op_constraint[key] < device[key]:
                    return False
            else:
                if op_constraint[key] != device[key]:
                    return False

        if key in device['labels']:
            if isinstance(value, int) or isinstance(value, float):
                if op_constraint[key] < device['labels'][key]:
                    return False
            else:
                if op_constraint[key] != device['labels'][key]:
                    return False

    return True

def make_decision(operators: List[Dict], devices: List[Dict]) -> Dict:
    variables = list(range(len(operators)))  # 应用变量列表
    domains = [list(range(len(devices))) for _ in operators]  # 取值域列表

    for variable in variables:
        # 获取当前变量对应的应用和部署条件
        app = operators[variable]
        domain = domains[variable]

        # 过滤满足部署条件的机器
        filtered_devices = [device_id for device_id in domain if check_constraints(app, devices[device_id])]

        # 更新取值域为满足条件的机器集合
        domains[variable] = filtered_devices

    # 使用回溯算法搜索解空间
    solution = backtrack({}, variables, domains, devices)

    solution_final = []

    for op_id, device_id in solution.items():
        app_name = operators[op_id]['name']
        machine_name = devices[device_id]['hostname']
        solution_final.append({
            "operator": app_name,
            "device": machine_name
        })

    return solution_final

def backtrack(solution: Dict, variables: List[int], domains: List[List[int]], machines: List[Dict]) -> Dict:
    """回溯算法搜索解空间"""
    if len(solution) == len(variables):
        return solution  # 找到解

    variable = select_unassigned_variable(variables, solution)
    for value in order_domain_values(variable, domains):
        if is_consistent(variable, value, solution, variables, domains):
            solution[variable] = value
            result = backtrack(solution, variables, domains, machines)
            if result is not None:
                return result
            del solution[variable]
    return None

def select_unassigned_variable(variables: List[int], solution: Dict) -> int:
    """选择未分配的变量"""
    for variable in variables:
        if variable not in solution:
            return variable

def order_domain_values(variable: int, domains: List[List[int]]) -> List[int]:
    """sort devices by heuristic rules"""
    return domains[variable]

def is_consistent(variable: int, value: int, solution: Dict, variables: List[int], domains: List[List[int]]) -> bool:
    """检查变量取值的一致性"""
    partial_solution = solution.copy()
    partial_solution[variable] = value

    for i, variable in enumerate(variables):
        if variable not in partial_solution:
            domain = [value for value in domains[i] if is_valid_assignment(variable, value, partial_solution, variables, domains)]
            if not domain:
                return False
    return True

def is_valid_assignment(variable: int, value: int, solution: Dict, variables: List[int], domains: List[List[int]]) -> bool:
    """检查变量取值是否有效"""
    app = _apps[variable]
    machine = _machines[value]
    return check_constraints(app, machine)

# sample data
# apps = [
#   {
#     "name": "picamera",
#     "constraints": {
#       "name": "k8s-node4",
#     }
#   },
#   {
#     "name": "imgaugment",
#     "constraints": {
#       "name": "k8s-node2"
#     }
#   },
#   {
#     "name": "humandetection",
#     "constraints": {
#         "arch": "arm64"
#     }
#   },
#   {
#     "name": "temperature",
#     "constraints": {
#       "sensor": "temp"
#     }
#   }
# ]
#
# machines = [{"name": "k8s-master", "ip": "172.22.152.175", "arch": "arm", "os": "Raspbian GNU/Linux 10 (buster)", "CPU": 4, "Memory": "1761928Ki", "labels": {"beta.kubernetes.io/arch": "arm", "beta.kubernetes.io/os": "linux", "id": "D", "kubernetes.io/arch": "arm", "kubernetes.io/hostname": "k8s-master", "kubernetes.io/os": "linux", "node-role.kubernetes.io/control-plane": "", "node.kubernetes.io/exclude-from-external-load-balancers": ""}, "location": None},
#             {"name": "k8s-node1", "ip": "172.22.22.58", "arch": "arm", "os": "Raspbian GNU/Linux 10 (buster)", "CPU": 4, "Memory": "1761928Ki", "labels": {"beta.kubernetes.io/arch": "arm", "beta.kubernetes.io/os": "linux", "id": "C", "kubernetes.io/arch": "arm", "kubernetes.io/hostname": "k8s-node1", "kubernetes.io/os": "linux", "sensor": "temp", "weatherhat": "true"}, "location": None},
#             {"name": "k8s-node2", "ip": "172.22.68.254", "arch": "arm", "os": "Raspbian GNU/Linux 10 (buster)", "CPU": 4, "Memory": "1761928Ki", "labels": {"beta.kubernetes.io/arch": "arm", "beta.kubernetes.io/os": "linux", "id": "A", "kubernetes.io/arch": "arm", "kubernetes.io/hostname": "k8s-node2", "kubernetes.io/os": "linux"}, "location": None},
#             {"name": "k8s-node3", "ip": "172.22.128.155", "arch": "arm64", "os": "Ubuntu 22.04.2 LTS", "CPU": 2, "Memory": "3903008Ki", "labels": {"arch": "arm64", "beta.kubernetes.io/arch": "arm64", "beta.kubernetes.io/os": "linux", "id": "B", "kubernetes.io/arch": "arm64", "kubernetes.io/hostname": "k8s-node3", "kubernetes.io/os": "linux"}, "location": None},
#             {"name": "k8s-node4", "ip": "172.22.85.72", "arch": "arm", "os": "Raspbian GNU/Linux 10 (buster)", "CPU": 4, "Memory": "1761928Ki", "labels": {"beta.kubernetes.io/arch": "arm", "beta.kubernetes.io/os": "linux", "kubernetes.io/arch": "arm", "kubernetes.io/hostname": "k8s-node4", "kubernetes.io/os": "linux"}, "location": None}]

#解决应用部署问题
# solution = deploy_apps(apps, machines)

#输出结果
# for app_id, machine_id in solution.items():
#     app_name = apps[app_id]['name']
#     machine_name = machines[machine_id]['name']
#     print(f"{app_name} 部署在 {machine_name}")