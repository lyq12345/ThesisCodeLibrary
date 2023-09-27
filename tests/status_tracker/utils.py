import json
import os

def get_device_status(filename):
    with open(filename, 'r') as json_file:
        data = json.load(json_file)
    return data


def get_operator_info(filename):
    with open(filename, 'r') as json_file:
        data = json.load(json_file)
    return data


def get_deployment_status(filename):
    with open(filename, 'r') as json_file:
        data = json.load(json_file)
    return data