import argparse
import requests
import time
import cv2
import os
import csv
# 解析命令行参数
parser = argparse.ArgumentParser(description='Send HTTP request to a target IP and port with specified rate')
parser.add_argument('--url', type=str, help='Target address')
parser.add_argument('--workflow', type=str, help='workflow id')
parser.add_argument('--object', type=str, help='The object type')
parser.add_argument('--rate', type=float, help='Request rate (requests per second)')
parser.add_argument('--num', type=int, help='the workflow number')
args = parser.parse_args()

def resize_img(img, new_width):
    aspect_ratio = float(new_width) / img.shape[1]
    new_height = int(img.shape[0] * aspect_ratio)
    resized_image = cv2.resize(img, (new_width, new_height))
    return resized_image

# 构造请求 URL
url = args.url
object_type = args.object
interval = 1 / float(args.rate)
# 发送 HTTP 请求
if object_type == "human":
    img_file = "test_human_2.jpeg"
elif object_type == "pose":
    img_file = "test_pose.jpg"
else:
    img_file = "test_fire.jpg"
# img_file = "test_human_2.jpeg" if operator == "human" else "test_fire.jpg"
data = [["res_time"]]
sum_res_time = 0.0

# read the image
image = cv2.imread(img_file)
if image is None:
    print("Cannot load image")
new_width = 640
resized_img = resize_img(image, new_width)
# lower the quality
quality = 65  # quality set to 65%
encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
_, jpeg_frame = cv2.imencode('.jpg', resized_img, encode_params)

save_folder = f"{args.num}_workflows"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

save_file = f"{save_folder}/real_delay_{args.workflow}.csv"
epoch = 3
count = 0
try:
    while count <= epoch:
        try:
            response = requests.post(url, data=jpeg_frame.tobytes(),
                                     headers={'Content-Type': 'image/jpeg'})
            if response.status_code == 200:
                result = response.json()
                response_time = response.elapsed.total_seconds()
                proc_time = round(float(result['process_time']), 4)
                print(f"request succeeded, response time: {response_time}, process time: {proc_time}")
                data.append([response_time])
                count += 1
            else:
                print(f"failed，status code: {response.status_code}")
                count += 1
        except Exception as e:
            print(f"failed，error: {str(e)}")
            count += 1
        time.sleep(interval)
except KeyboardInterrupt:
    print("Request sending stopped by user")

# avg_time = sum_res_time / len(data["res_time"])
with open(save_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)