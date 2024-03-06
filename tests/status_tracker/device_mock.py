import random
import json
raspi_model = {
        "id": 0,
        "hostname": "node1",
        "model": "raspberrypi-4b",
        "ip": "172.22.152.175",
        "arch": "arm64",
        "os": "Raspbian GNU/Linux 10 (buster)",
        "resources": {
          "system": {
             "cpu": 400,
            "gpu": 0,
            "memory": 4096,
            "storage": 131072
          },
          "hardware": None
        },
        "power": "unlimited",
        "location": 100
  }
jetson_nano_model = {
        "id": 0,
        "hostname": "node5",
        "model": "jetson-nano",
        "ip": "172.22.152.175",
        "arch": "arm64",
        "os": "ubuntu 18.04",
        "resources": {
          "system": {
            "cpu": 400,
            "gpu": 1,
            "memory": 4096,
            "storage": 65600
          },
          "hardware": None

        },
        "power": "unlimited",
        "location": 100
    }
jetson_xavier_model = {
        "id": 0,
        "hostname": "node3",
        "model": "jetson-xavier",
        "ip": "172.22.152.175",
        "arch": "arm64",
        "os": "ubuntu 18.04",
        "resources": {
            "system": {
                "cpu": 800,
                "gpu": 1,
                "memory": 32768,
                "storage": 65600
            },
            "hardware": None

        },
        "power": "unlimited",
        "location": 100
    }

sensor_model = [
        {
          "id": 1,
          "sensor": "IMX519",
            "rate": 0.02,
            "size": 1200
        }
    ]

def generate_random_ipv4():
    ip_parts = [str(random.randint(0, 255)) for _ in range(4)]
    return ".".join(ip_parts)

def create_from_model(id, model, sensor, power):
    device_data = {}
    if model == "raspberrypi-4b":
        device_data = raspi_model
    elif model == "jetson-nano":
        device_data = jetson_nano_model
    elif model == "jetson-xavier":
        device_data = jetson_xavier_model

    device_data["id"] = id
    device_data["hostname"] = f"node{id}"
    if sensor:
        device_data["resources"]["hardware"] = get_sensor_model()
        device_data["resources"]["hardware"][0]["id"] = id
    else:
        device_data["resources"]["hardware"] = []

    device_data["power"] = power
    return json.dumps(device_data)

def allocate_types(num_devices):
    categories = ['edge-server', 'IoT-insitu', 'IoT-mobile']
    weights = [0.6, 0.3, 0.1]
    allocated_categories = random.choices(categories, weights=weights, k=num_devices)
    return allocated_categories

def get_random_coordinate():
    x = random.uniform(-20, 80)
    y = random.uniform(-20, 60)
    z = random.uniform(0, 60)
    return [x, y, z]

def get_sensor_model():
    sensor_template = sensor_model
    sensor_template[0]["rate"] = random.uniform(0.02, 0.1)
    return sensor_template

def generate_devices(num_devices):

    device_list = []
    edge_models = ["raspberrypi-4b", "jetson-nano", "jetson-xavier"]
    weight1 = [0.2, 0.3, 0.5]
    iot_models = ["raspberrypi-4b", "jetson-nano"]
    weight2 = [0.5, 0.5]

    device_types = allocate_types(num_devices)

    # edge server; IoT devices(in-situ); IoT devices(mobile)
    for id in range(num_devices):
        device_type = device_types[id]
        if device_type == "edge-server":
            device_model = random.choices(edge_models, weights=weight1, k=1)[0]
        else:
            device_model = random.choices(iot_models, weights=weight2, k=1)[0]

        sensor = False
        power = float('inf')
        if device_type != "edge-server":
            sensor = True
            if device_type == "IoT-mobile":
                power = random.randint(1820, 3000)
        device = json.loads(create_from_model(id, device_model, sensor, power))

        initial_location = get_random_coordinate()
        device["location"] = initial_location
        device["type"] = device_type
        # print(device)
        device_list.append(device)

    return device_list

# devices = generate_devices(20)
# print(devices)




