#!/bin/bash

# 获取进程 ID
pid="$1"
freq="$2"
# 如果找不到进程，则提示用户并退出
if [ -z "$pid" ]; then
    echo "Process '$process_name' not found."
    exit 1
fi

for ((i=1; i<=20; i++)); do
	# 使用 top 命令获取进程的 CPU 占用率
	cpu_usage=$(top -n 1 -b -p "$pid" | grep "$pid" | awk '{print $9}')
	echo "$cpu_usage" >> top_cpu_$freq.log
	sleep 1	
done

echo "CPU usage of process '$process_name' (PID: $pid): $cpu_usage%"

