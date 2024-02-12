import random

object_dict = {
    "human": 1,
    "fire": 2
}

def generate_tasks(num_tasks, device_list):
    sensor_list = []
    for dev in device_list:
        if len(dev["resources"]["hardware"]) > 0:
            sensor_list.append(dev["id"])
    random_selection = random.sample(sensor_list, num_tasks)
    # print(random_selection)

    task_list = []
    count = 0

    for sensor_id in random_selection:
        object = random.choice(["human", "fire"])
        object_code = object_dict[object]
        delay = random.uniform(1.0, 10.0)
        # print(delay)
        rate = device_list[sensor_id]["resources"]["hardware"][0]["rate"]
        size = device_list[sensor_id]["resources"]["hardware"][0]["size"]
        data = {"id": count, "source": sensor_id, "rate": rate, "size": size, "object": object, "object_code": object_code, "delay": delay, "priority": 10}
        count += 1
        task_list.append(data)

    return task_list

