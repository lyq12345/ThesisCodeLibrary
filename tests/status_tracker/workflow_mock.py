import random

object_dict = {
    "human": 1,
    "fire": 2
}

workflow_templates = [
    # [4, 1, 7],
    # [4, 1, 8],
    # [5, 2, 6],
    # [1, 2]
    # [3, 9]
    # [1],
    # [2],
    [1, 2],
    [1, 2, 1, 7],
    [5, 2, 6],
    [4, 1, 7],
    [4, 1, 8, 4]
]

# workflow_templates2 = [
#     [1],
#     [2]
# ]


def random_weights(n):
    # 生成 n 个随机权值
    weights = [random.random() for _ in range(n)]

    # 对权值进行归一化处理
    total = sum(weights)
    weights = [w / total for w in weights]

    return weights
def generate_workflows(num_workflows, device_list):
    sensor_list = []
    for dev in device_list:
        if len(dev["resources"]["hardware"]) > 0:
            sensor_list.append(dev["id"])

    final_workflows = []
    priorities = random_weights(num_workflows)

    # workflow_list = []
    # for _ in range(num_workflows):
    #     sample = random.choice(workflow_templates)
    #     workflow_list.append(sample)


    for i in range(num_workflows):
        workflow = random.choice(workflow_templates)
        delay = random.uniform(5.0, 15.0)
        sensor_id = random.choice(sensor_list)
        rate = device_list[sensor_id]["resources"]["hardware"][0]["rate"]
        size = device_list[sensor_id]["resources"]["hardware"][0]["size"]
        data = {"id": i, "source": sensor_id, "rate": rate, "size": size, "workflow": workflow, "delay": delay, "pri": priorities[i]}
        final_workflows.append(data)

    return final_workflows

# if __name__ == '__main__':
#     generate_workflows(10, )

