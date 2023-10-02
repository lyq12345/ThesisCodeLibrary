import json
from id_generator import generate_operator_id
class Greedy_Decider:
    # def __init__(self):
    #     self.devices = self.read_json()

    def read_json(self, filename):
        with open(filename, 'r') as json_file:
            data = json.load(json_file)
        return data
    def is_hardware_consistent(self, hardware_resources, hardware_requirements):
        if hardware_requirements is None:
            return True
        if hardware_requirements is not None and hardware_resources is None:
            return False
        for requirement in hardware_requirements:
            flag = False
            for resource in hardware_resources:
                if resource["sensor"] == requirement["sensor"]:
                    flag = True
            if not flag:
                return False
        return True
    def is_system_consistent(self, system_resources, system_requirements):
        for key, value in system_requirements.items():
            if key not in system_resources:
                return False
            if key in system_resources:
                if isinstance(value, int) or isinstance(value, float):
                    if system_resources[key] < system_resources[key]:
                        return False
                else:
                    if system_requirements[key] != system_resources[key]:
                        return False

        return True

    def deploy_operator(self, device, operator):
        operator_name = operator["name"]
        device_name = device["hostname"]
        print(f"operator {operator_name} is deployed on device {device_name}")

    def resource_allocation(self, device, operator):
        for dev in self.devices:
            if dev["hostname"] == device["hostname"]:
                # resource consumption
                for key, value in operator["requirements"]["system"].items():
                    dev["resources"]["system"][key] -= value

    def allocation_withdraw(self, device, operator):
        for dev in self.devices:
            if dev["hostname"] == device["hostname"]:
                # resource consumption
                for key, value in operator["requirements"]["system"].items():
                    dev["resources"]["system"][key] += value
    def match_operators_with_devices(self, operator_pairs, devices):

        self.devices = devices

        # store the solution
        solutions = []

        # traverse the operator pairs
        for pair in operator_pairs:
            source_operator = pair['source']
            processing_operator = pair['processing']

            source_system_requirements = source_operator['requirements']['system']
            source_hardware_requirements = source_operator['requirements']['hardware']

            processing_system_requirements = processing_operator['requirements']['system']
            processing_hardware_requirements = processing_operator['requirements']['hardware']

            # try to map a device for source operator
            source_matched_device = None

            # sort devices by available memory size
            for dev in sorted(self.devices, key=lambda x: x["resources"]["system"]["memory"], reverse=True):
                system_resources = dev['resources']['system']
                hardware_resources = dev['resources']['hardware']
                if self.is_hardware_consistent(hardware_resources, source_hardware_requirements) \
                        and self.is_system_consistent(system_resources, source_system_requirements):
                    source_matched_device = dev
                    self.resource_allocation(source_matched_device, source_operator)
                    break  # found suitable device

            if source_matched_device is None:
                break

            # try to map a device for processing operator
            processing_matched_device = None
            for dev in sorted(self.devices, key=lambda x: x["resources"]["system"]["memory"], reverse=True):
                system_resources = dev['resources']['system']
                hardware_resources = dev['resources']['hardware']
                if self.is_hardware_consistent(hardware_resources, processing_hardware_requirements) \
                        and self.is_system_consistent(
                        system_resources, processing_system_requirements):
                    processing_matched_device = dev
                    self.resource_allocation(processing_matched_device, processing_operator)
                    break

            if processing_matched_device is None:
                self.allocation_withdraw(source_matched_device, source_operator)
                break


            # only if source and processing operators are both satisfied
            self.deploy_operator(source_matched_device, source_operator)
            self.deploy_operator(processing_matched_device, processing_operator)
            # source_id = generate_operator_id()
            # process_id = generate_operator_id()
            data_flow = {
                "source": {
                    "operator": source_operator["name"],
                    "device": source_matched_device["hostname"]
                },
                "processing": {
                    "operator": processing_operator["name"],
                    "device": processing_matched_device["hostname"]
                }
            }
            solutions.append(data_flow)


        return solutions



    # matches = match_operators_with_devices(operator_pairs, devices)
    # print("匹配结果:")
    # for (source_operator, processing_operator), (source_device, processing_device) in matches.items():
    #     print(f"Source Operator: {source_operator}, Source Device: {source_device}")
    #     print(f"Processing Operator: {processing_operator}, Processing Device: {processing_device}")
