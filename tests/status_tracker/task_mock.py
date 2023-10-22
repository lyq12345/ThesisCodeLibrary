import random

object_dict = {
    "human": 1,
    "fire": 2
}

def generate_tasks(num_tasks, device_list):
    sensors_list = list(range(len(device_list)))
    random_selection = random.sample(sensors_list, num_tasks)
    # print(random_selection)

    task_list = []
    count = 0

    for sensor in random_selection:
        object = random.choice(["human", "fire"])
        object_code = object_dict[object]
        delay = random.uniform(5.0, 20.0)
        # print(delay)
        data = {"id": count, "source": sensor, "object": object, "object_code": object_code, "delay": delay, "priority": 10}
        count += 1
        task_list.append(data)

    return task_list

