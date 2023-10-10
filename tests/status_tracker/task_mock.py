import random


def generate_tasks(num_tasks, device_list):
    sensors_list = list(range(len(device_list)))
    random_selection = random.sample(sensors_list, num_tasks)
    print(random_selection)

    task_list = []
    count = 0

    for sensor in random_selection:
        data = {"id": count, "source": sensor, "object": "human", "object_code": 1, "delay": 10, "priority": 10}
        count += 1
        task_list.append(data)

    return task_list

