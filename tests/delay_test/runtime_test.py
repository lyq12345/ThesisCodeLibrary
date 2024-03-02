import cv2
import requests
import time
import pandas as pd
import sys
from datetime import datetime

# import keyboard

# image_path = 'fire.jpg'
devices_urls = {
    # "pi": "172.31.158.52",
    "nano": "128.200.218.98",
    "xavier": "128.200.218.112"
}

interval = 2

def resize_img(img, new_width):
    aspect_ratio = float(new_width) / img.shape[1]
    new_height = int(img.shape[0] * aspect_ratio)
    resized_image = cv2.resize(img, (new_width, new_height))
    return resized_image

def process_distribution(device, operator, version, port, interval, epoth):
    if operator == "human":
        img_file = "test_human_2.jpeg"
    elif operator == "pose":
        img_file = "test_pose.jpg"
    else:
        img_file = "test_fire.jpg"
    # img_file = "test_human_2.jpeg" if operator == "human" else "test_fire.jpg"
    target_url = f"http://{devices_urls[device]}:{port}/process_video"
    data = {'proc_time': []}

    # read the image
    image = cv2.imread(img_file)
    if image is None:
        print("Cannot load image")
        return
    new_width = 640
    resized_img = resize_img(image, new_width)
    # lower the quality
    quality = 65  # quality set to 65%
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, jpeg_frame = cv2.imencode('.jpg', resized_img, encode_params)

    if epoth == 0:
        while True:
            try:
                response = requests.post(target_url, data=jpeg_frame.tobytes(),
                                         headers={'Content-Type': 'image/jpeg'})
                if response.status_code == 200:
                    result = response.json()
                    proc_time = round(float(result['process_time']), 4)
                    print(f"request succeeded, time: {proc_time}")
                    data['proc_time'].append(proc_time)
                else:
                    print(f"failed，status code: {response.status_code}")
            except Exception as e:
                print(f"failed，error: {str(e)}")
            time.sleep(interval)

    else:
        for i in range(epoth):
            # send image through http
            # if keyboard.is_pressed('q'):
            #     print("'q' pressed, exit.")
            #     break
            try:
                response = requests.post(target_url, data=jpeg_frame.tobytes(),
                                         headers={'Content-Type': 'image/jpeg'})
                if response.status_code == 200:
                    result = response.json()
                    proc_time = round(float(result['process_time']), 4)
                    print(f"request {i+1} succeeded, time: {proc_time}")
                    data['proc_time'].append(proc_time)
                else:
                    print(f"round {i + 1} failed，status code: {response.status_code}")
            except Exception as e:
                print(f"round {i + 1} failed，error: {str(e)}")
            time.sleep(interval)

        # only collect reverse 50
        if epoth > 50:
            data['proc_time'] = data['proc_time'][-50:]
        avg_delay = sum(data['proc_time']) / len(data['proc_time'])
        data['proc_time'].append(avg_delay)
        current_time = datetime.now()
        timestamp_format = current_time.strftime("%y%m%d%H%M%S")
        save_file = f"results/qos_tests/power/{device}_{operator}_{version}_{interval}_{timestamp_format}.csv"
        df = pd.DataFrame(data)
        df.to_csv(save_file, index=False)  # 保存到CSV文件
        print("Completed.")

if len(sys.argv) > 1:
    device = sys.argv[1]
    operator = sys.argv[2]
    version = sys.argv[3]
    port = sys.argv[4]
    interval = float(sys.argv[5])
    epoth = int(sys.argv[6])
    process_distribution(device, operator, version, port, interval, epoth)
