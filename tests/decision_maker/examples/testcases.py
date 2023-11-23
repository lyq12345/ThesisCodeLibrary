import numpy as np
def generate_testcase():
    device_list = [
        {'id': 0, 'hostname': 'node0', 'model': 'raspberrypi-4b', 'ip': '172.22.152.175', 'arch': 'arm64', 'os': 'Raspbian GNU/Linux 10 (buster)',
         'resources': {'system': {'cpu': 4, 'gpu': 0, 'memory': 4096, 'storage': 131072}, 'hardware': [{'id': 0, 'sensor': 'IMX519', 'rate': 0.483, 'size': 1200}]}, 'power': 1000000, 'location': 100},
        {'id': 1, 'hostname': 'node1', 'model': 'raspberrypi-4b', 'ip': '172.22.152.175', 'arch': 'arm64', 'os': 'Raspbian GNU/Linux 10 (buster)',
         'resources': {'system': {'cpu': 4, 'gpu': 0, 'memory': 4096, 'storage': 131072}, 'hardware': [{'id': 1, 'sensor': 'IMX519', 'rate': 0.483, 'size': 1200}]}, 'power': 1000000, 'location': 100},
        {'id': 2, 'hostname': 'node2', 'model': 'jetson-xavier', 'ip': '172.22.152.175', 'arch': 'arm64', 'os': 'ubuntu 18.04',
         'resources': {'system': {'cpu': 4, 'gpu': 1, 'memory': 32768, 'storage': 65600}, 'hardware': [{'id': 2, 'sensor': 'IMX519', 'rate': 0.483, 'size': 1200}]}, 'power': 1820, 'location': 100},
        {'id': 3, 'hostname': 'node3', 'model': 'raspberrypi-4b', 'ip': '172.22.152.175', 'arch': 'arm64', 'os': 'Raspbian GNU/Linux 10 (buster)',
         'resources': {'system': {'cpu': 4, 'gpu': 0, 'memory': 4096, 'storage': 131072}, 'hardware': [{'id': 3, 'sensor': 'IMX519', 'rate': 0.483, 'size': 1200}]}, 'power': 1820, 'location': 100},
        {'id': 4, 'hostname': 'node4', 'model': 'raspberrypi-4b', 'ip': '172.22.152.175', 'arch': 'arm64', 'os': 'Raspbian GNU/Linux 10 (buster)',
         'resources': {'system': {'cpu': 4, 'gpu': 0, 'memory': 4096, 'storage': 131072}, 'hardware': [{'id': 4, 'sensor': 'IMX519', 'rate': 0.483, 'size': 1200}]}, 'power': 1820, 'location': 100}]


    task_list = [
        {'id': 0, 'source': 4, 'rate': 0.483, 'size': 1200, 'object': 'fire', 'object_code': 2, 'delay': 6.623139336962066, 'priority': 10},
        {'id': 1, 'source': 0, 'rate': 0.483, 'size': 1200, 'object': 'human', 'object_code': 1, 'delay': 6.15679301502469, 'priority': 10},
        {'id': 2, 'source': 1, 'rate': 0.483, 'size': 1200, 'object': 'human', 'object_code': 1, 'delay': 2.025443695770312, 'priority': 10}
    ]


    propagation_matrix = np.array([[0,         1.6927156,  2.92235022, 3.51951305, 4.86682215],
                             [1.6927156,  0.         ,1.96944883, 4.59353894, 1.14099295],
                             [2.92235022, 1.96944883, 0.         ,3.26888801, 1.83675882],
                             [3.51951305, 4.59353894, 3.26888801, 0.        , 1.94153141],
                             [4.86682215, 1.14099295, 1.83675882, 1.94153141, 0.        ]])

    return device_list, task_list, propagation_matrix