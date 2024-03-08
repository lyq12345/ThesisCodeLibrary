import json
import paramiko

hostnames = ["edge1", "edge2", "edge3", "iot1", "iot2"]
memorys = [32, 4, 2, 2, 2]
information = {
    "edge1": {
        "username": "xavier",
        "password": "DSM12345",
        "ip": "128.200.218.112"
    },
    "edge2": {
            "username": "nano",
            "password": "DSM12345",
            "ip": "128.195.53.178 "
        },
    "edge3": {
            "username": "pi",
            "password": "DSM12345",
            "ip": "172.31.183.184"
        },
    "iot1": {
            "username": "pi",
            "password": "DSM12345",
            "ip": "172.31.245.17"
        },
    "iot2": {
            "username": "pi",
            "password": "DSM12345",
            "ip": "172.31.183.180"
        },
}

def deploy_operator(hostname, operators):
    port = 22

    # 创建 SSH 客户端实例
    client = paramiko.SSHClient()

    # 忽略主机密钥检查
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    username = information[hostname]["username"]
    password = information[hostname]["password"]

    try:
        # 连接到远程服务器
        client.connect(hostname, port, username, password)

        for operator in operators:
            op_id = operator[0]
            image = operator[1]
            wf_id = operator[2]
            object_type = operator[3]
            port = 8848 if object_type == "human" else 8849
            host_port = 40000 + int(op_id)
            command = f"docker run --name operator_{op_id} --rm -d -p {host_port}:{port} {image}"
            # 执行命令并获取输出
            stdin, stdout, stderr = client.exec_command(command)

            # 打印命令输出
            container_id = stdout.read().decode()
            print(container_id)

            # print(stderr.read().decode())

    finally:
        client.close()

def trigger_workflows(workflow_list, workflow_url_mapping, solver):
    source_workflow_mapping = {}
    for wf_id, workflow in enumerate(workflow_list):
        source_dev_id = workflow["source"]
        rate = workflow["rate"]
        url = workflow_url_mapping[wf_id]
        service_code = workflow["workflow"][0]
        object_type = "fire" if service_code == 2 else "human"
        if source_dev_id not in source_workflow_mapping.keys():
            source_workflow_mapping[source_dev_id] = []
        source_workflow_mapping[source_dev_id].append([wf_id, url, rate, object_type])

    for source_dev_id, requests in source_workflow_mapping.items():
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        hostname = hostnames[source_dev_id]
        username = information[hostname]["username"]
        password = information[hostname]["password"]
        try:
            client.connect(hostname, 22, username, password)

            for request in requests:
                wf_id = request[0]
                url = request[1]
                rate = request[2]
                object_type = request[3]
                command = f"nohup python send_request.py --url {url} --object {object_type} --rate {rate} --num {len(workflow_list)} --workflow {wf_id} --solver {solver} &"
                # 执行命令并获取输出
                stdin, stdout, stderr = client.exec_command(command)

        finally:
            client.close()

def calculate_resource():
    cpu_sum = 0.0
    memory_sum = 0.0
    cpu_max = 2400
    memory_max = 42
    for idx, hostname in enumerate(hostnames):
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        username = information[hostname]["username"]
        password = information[hostname]["password"]
        try:
            # print(hostname)
            cpu_sum = 0.0
            memory_sum = 0.0
            # execute 10 times and get average
            client.connect(hostname, 22, username, password)
            command = "python3 get_cpu_memory.py 2>&1"
            for _ in range(10):
                # 执行命令并获取输出
                stdin, stdout, stderr = client.exec_command(command)
                output = stdout.read().decode()
                # print(output)
                cpu = float(output.split("-")[0])
                memory = float(output.split("-")[1])
                print(hostname, cpu, memory)
                cpu_sum += cpu
                memory_sum += memory
            cpu_avg = cpu_sum / 10
            cpu_memory = memory_sum / 10
            # print(cpu)
            # print(memory)
            cpu_sum += cpu_avg
            memory_sum += memorys[idx]*cpu_memory

        finally:
            client.close()
    cpu_usage = cpu_sum / cpu_max
    memory_usage = memory_sum / memory_max
    return [cpu_usage, memory_usage]
# def calculate_power():

def kill_threads_and_containers():
    for idx, hostname in enumerate(hostnames):
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        username = information[hostname]["username"]
        password = information[hostname]["password"]
        try:
            # 连接到远程服务器
            client.connect(hostname, 22, username, password)

            # 连续执行两个命令
            commands = ["./reset_containers.sh", "pkill -f python"]
            for command in commands:
                try:
                    stdin, stdout, stderr = client.exec_command(command)
                    output = stdout.read().decode()
                except paramiko.SSHException as e:
                    print(f"Error executing command '{command}': {e}")
                    # 如果第一个命令报错，则跳过后续命令
        finally:
            # 关闭 SSH 连接
            client.close()
    print("Kill Finished")


# calculate_resource()
# connect_remote("edge1", "xavier", "DSM12345")
