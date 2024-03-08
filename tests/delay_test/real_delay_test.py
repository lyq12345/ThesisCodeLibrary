import subprocess
import paramiko
# 设备列表
hosts = ['edge1', 'edge2', 'edge3', 'iot1', 'iot2']
ips = ['128.200.218.112', '128.195.53.178', '172.31.183.184', '172.31.245.17', '172.31.183.180']
port = 22

information = {
    "edge1": {
        "username": "xavier",
        "password": "DSM12345"
    },
    "edge2": {
            "username": "nano",
            "password": "DSM12345"
        },
    "edge3": {
            "username": "pi",
            "password": "DSM12345"
        },
    "iot1": {
            "username": "pi",
            "password": "DSM12345"
        },
    "iot2": {
            "username": "pi",
            "password": "DSM12345"
        },
}

for i, host in enumerate(hosts):
    # 创建 SSH 客户端实例
    client = paramiko.SSHClient()

    # 忽略主机密钥检查
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    host_ip = ips[i]
    username = information[host]["username"]
    password = information[host]["password"]
    try:
        # 连接到设备1
        client.connect(host_ip, port, username, password)

        for j, other_host in enumerate(hosts):
            # 执行命令并获取输出
            other_ip = ips[j]
            stdin, stdout, stderr = client.exec_command(f'ping -c 5 {other_ip}')
            print(f"Latency between {host} and {other_host}:")
            # 打印命令输出
            print(stdout.read().decode())
            print(stderr.read().decode())

    finally:
        # 关闭 SSH 连接
        client.close()
