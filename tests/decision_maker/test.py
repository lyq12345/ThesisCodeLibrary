import random

# Sample data: service details, device storage, latencies, etc.
services = {
    'Service1': {'storage': 5},
    'Service2': {'storage': 7},
    # Add more services and their storage consumption here
}

devices = {
    'Device1': {'capability': 15},
    'Device2': {'capability': 20},
    # Add more devices and their storage capabilities here
}

latencies = {
    ('Device1', 'Device2'): 10,
    ('Device1', 'Device3'): 15,
    # Define latency between each pair of devices
}

# Function to calculate total latency given service placements
def calculate_total_latency(placement):
    total_latencies = 0
    for service, device in placement.items():
        total_latencies += latencies[(device, devices[device]['assigned_service'])]
    return total_latencies

# Function to check if a device can accommodate a service
def can_deploy(service, device):
    return (devices[device]['capability'] >= services[service]['storage'])

# Function to generate initial random placement of services on devices
def generate_initial_placement():
    placement = {}
    available_devices = list(devices.keys())
    for service in services:
        random_device = random.choice(available_devices)
        while not can_deploy(service, random_device):
            available_devices.remove(random_device)
            if not available_devices:
                raise ValueError("No available devices to deploy service")
            random_device = random.choice(available_devices)
        placement[service] = random_device
        devices[random_device]['capability'] -= services[service]['storage']
        devices[random_device]['assigned_service'] = service
        available_devices.remove(random_device)
    return placement

# Iterated Local Search algorithm
def iterated_local_search(iterations):
    current_placement = generate_initial_placement()
    best_placement = current_placement.copy()
    best_avg_latency = calculate_total_latency(best_placement) / len(services)

    for i in range(iterations):
        # Generate neighboring solution by randomly swapping services between devices
        random_service_1, random_service_2 = random.sample(services.keys(), 2)
        new_placement = current_placement.copy()
        new_placement[random_service_1], new_placement[random_service_2] = (
            new_placement[random_service_2],
            new_placement[random_service_1],
        )

        # Calculate total latency of the new solution
        new_total_latency = calculate_total_latency(new_placement)
        new_avg_latency = new_total_latency / len(services)

        # If the new solution is better, accept it as the current solution
        if new_avg_latency < best_avg_latency:
            best_placement = new_placement.copy()
            best_avg_latency = new_avg_latency

        current_placement = new_placement.copy()

    return best_placement, best_avg_latency

best_placement, best_avg_latency = iterated_local_search(1000)

print("Best Placement:", best_placement)
print("Best Average Latency:", best_avg_latency)
