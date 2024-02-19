import random

object_dict = {
    "human": 1,
    "fire": 2
}

workflow_templates = [
    [4, 1, 7],
    [4, 1, 8],
    [5, 2, 6],
    [3, 9]
]

def generate_workflows(num_workflows, device_list):
    sensor_list = []
    for dev in device_list:
        if len(dev["resources"]["hardware"]) > 0:
            sensor_list.append(dev["id"])

    final_workflows = []

    # workflow_list = []
    # for _ in range(num_workflows):
    #     sample = random.choice(workflow_templates)
    #     workflow_list.append(sample)

    for i in range(num_workflows):
        workflow = random.choice(workflow_templates)
        delay = random.uniform(1.0, 10.0)
        sensor_id = random.choice(sensor_list)
        rate = device_list[sensor_id]["resources"]["hardware"][0]["rate"]
        size = device_list[sensor_id]["resources"]["hardware"][0]["size"]
        data = {"id": i, "source": sensor_id, "rate": rate, "size": size, "workflow": workflow, "delay": delay}
        final_workflows.append(data)

    return final_workflows

# if __name__ == '__main__':
#     generate_workflows(10, )

