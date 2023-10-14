import requests
import time

# 目标URL
url = 'https://example.com/api/endpoint'  # 替换成您要请求的接口URL

# 速率控制参数
requests_per_second = 2  # 每秒发送的请求数
interval = 1 / requests_per_second

# 主循环
while True:
    try:
        # 发送GET请求
        response = requests.get(url)
        if response.status_code == 200:
            print(f'Success: Request sent to {url}')
        else:
            print(f'Error: Request to {url} failed with status code {response.status_code}')
    except Exception as e:
        print(f'Error: {str(e)}')

    # 按指定速率休眠
    time.sleep(interval)
