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
             "cpu": 4,
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
            "cpu": 4,
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
                "cpu": 4,
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
            "rate": 0.4830,
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
        device_data["resources"]["hardware"] = sensor_model
        device_data["resources"]["hardware"][0]["id"] = id
    else:
        device_data["resources"]["hardware"] = []
    if power:
        device_data["power"] = float('inf')
    else:
        device_data["power"] = 1820
    return json.dumps(device_data)

def generate_devices(num_devices):
    device_list = []
    options = ["raspberrypi-4b", "jetson-nano", "jetson-xavier"]
    weights = [0.7, 0.2, 0.1]  # prob for each model
    for id in range(num_devices):
        model = random.choices(options, weights, k=1)[0]
        # sensor = random.choice([True, False])
        sensor = True
        power = random.choice([True, False])
        device = json.loads(create_from_model(id, model, sensor, power))
        # print(device)
        device_list.append(device)

    return device_list




