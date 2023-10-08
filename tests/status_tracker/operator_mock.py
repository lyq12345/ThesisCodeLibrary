import random

object_dict = {
    "human": 1,
    "fire": 2,
    "window": 3
}

def create_operator(id):
    object = random.choice(['human', 'fire', 'window'])
    level = random.choice([1, 2, 3])
    data = {
    "name": "joelee0515/firedetection:tinyyolov3-measure-time",
    "type": "processing",
    "object": object,
    "object_code":  object_dict[object],
    "version": "tinyyolov3",
    "accuracy": 0.3,
    "speed": 2,
    "power_consumption": 0,
    "requirements": {
      "system": {
        "cpu": 2,
        "gpu": 0,
        "storage": 763,
        "memory": 408
      },
      "hardware": None

    }
  }