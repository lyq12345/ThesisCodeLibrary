import requests
import json
import sys

# 定义docker swarm的管理节点ip,端口号，API版本，服务名， 服务URL
# 在后期集成到自动化部署时，需要精简数据结构，完善data, 增加精准判断及空间回收
# API更多用途参考: https://docs.docker.com/engine/api/v1.29/
docker_swarm_ip_port = "ip:port"
docker_swarm_api_version = "v1.29"
docker_swarm_service_name = "tomcatbv"

get_service_url = "http://%s/%s/services/%s" % (
docker_swarm_ip_port, docker_swarm_api_version, docker_swarm_service_name)

data = {
    "Name": "tomcatbv",
    "TaskTemplate": {
        "ContainerSpec": {
            "Image": "harbor_op/project_name/tomcat:20170620demo",
        },
        "Placement": {},
        "RestartPolicy": {
            "Condition": "on-failure",
            "Delay": 10000000000,
            "MaxAttempts": 10
        }
    },
    "Mode": {
        "Replicated": {
            "Replicas": 1
        }
    },
    "UpdateConfig": {
        "Parallelism": 2,
        "FailureAction": "pause",
    },
    "RollbackConfig": {
        "Parallelism": 1,
    },
    "Labels": {
        "foo": "bar"
    }
}


# 创建docker swarm service服务
def create_service():
    url = "http://%s/%s/services/create" % (docker_swarm_ip_port, docker_swarm_api_version)
    data_json = json.dumps(data)
    r = requests.post(url, data=data_json)
    print
    r.text
    if r.status_code == 201:
        print
        "created ok"
    else:
        print
        "create error"
        sys.exit()


# 更新docker swarm service服务
def update_service(version_index):
    url = "http://%s/%s/services/%s/update?version=%s" % (
    docker_swarm_ip_port, docker_swarm_api_version, docker_swarm_service_name, version_index)
    data_json = json.dumps(data)
    r = requests.post(url, data=data_json)
    print
    version_index, r.text, r.status_code
    if r.status_code == 200:
        print
        "updated ok"
    else:
        print
        "update error"
        sys.exit()


def main():
    get_service_url = "http://%s/%s/services/%s" % (
    docker_swarm_ip_port, docker_swarm_api_version, docker_swarm_service_name)
    r = requests.get(get_service_url)
    print
    get_service_url, r.status_code, r.text
    if r.status_code != 200:
        create_service()
    else:
        version_index = eval(r.text)["Version"]["Index"]
        update_service(version_index)


if __name__ == "__main__":
    main()