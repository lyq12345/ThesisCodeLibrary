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

    # the number of sensors should be the same as workflows
    sensor_random_selection = random.sample(sensor_list, num_workflows)
    final_workflows = []

    # workflow_list = []
    # for _ in range(num_workflows):
    #     sample = random.choice(workflow_templates)
    #     workflow_list.append(sample)

    count = 0

    # each sensor randomly pick up a sensor as data source
    for sensor_id in sensor_random_selection:
        workflow = random.choice(workflow_templates)
        delay = random.uniform(1.0, 10.0)
        # print(delay)
        rate = device_list[sensor_id]["resources"]["hardware"][0]["rate"]
        size = device_list[sensor_id]["resources"]["hardware"][0]["size"]
        data = {"id": count, "source": sensor_id, "rate": rate, "size": size, "workflow": workflow, "delay": delay}
        count += 1
        final_workflows.append(data)

    return final_workflows

# if __name__ == '__main__':
#     generate_workflows(10, )

