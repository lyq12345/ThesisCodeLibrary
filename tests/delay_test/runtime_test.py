import cv2
import requests
import time
import pandas as pd
import sys

# image_path = 'fire.jpg'
devices_urls = {
    # "pi": "172.31.158.52",
    # "nano": "172.31.249.146",
    "xavier": "128.200.218.112"
}


def resize_img(img, new_width):
    aspect_ratio = float(new_width) / img.shape[1]
    new_height = int(img.shape[0] * aspect_ratio)
    resized_image = cv2.resize(img, (new_width, new_height))
    return resized_image

def process_distribution(device, operator, version, epoth):
    port = 8848 if operator == "human" else 8849
    img_file = "test_human_2.jpeg" if operator == "human" else "test_fire.jpg"
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

    for i in range(epoth):
        # send image through http
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
        # sleep for 2 seconds
        time.sleep(2)
    save_file = f"results/raw_{device}_{operator}_{version}.csv"
    df = pd.DataFrame(data)
    df.to_csv(save_file, index=False)  # 保存到CSV文件
    print("Completed.")




# def process_average(device, operator):
#     port = 8848 if operator == "human" else 8849
#     img_file = "test_human_2.jpeg" if operator == "human" else "fire.jpg"
#     target_url = f"http://{devices_urls[device]}:{port}/process_video"
#
#     # read the image
#     image = cv2.imread(image_path)
#     if image is None:
#         print("Cannot load image")
#         return
#     new_width = 640
#     resized_img = resize_img(image, new_width)
#     # lower the quality
#     quality = 65  # quality set to 65%
#     encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
#     _, jpeg_frame = cv2.imencode('.jpg', resized_img, encode_params)
#     sum = 0.0
#     for i in range(n):
#         # 发送图像到目标端点
#         try:
#             response = requests.post(target_url, data=jpeg_frame.tobytes(),
#                                      headers={'Content-Type': 'image/jpeg'})
#             if response.status_code == 200:
#                 print(f"round {i+1} succeeded")
#                 result = response.json()
#                 proc_time = float(result['process_time'])
#                 sum += proc_time
#             else:
#                 print(f"round {i+1} failed，status code: {response.status_code}")
#         except Exception as e:
#             print(f"round {i+1} failed，error: {str(e)}")
#
#         # sleep for 2 seconds
#         time.sleep(2)
#     avg_time = sum / n
#     print(f"Average processing time over {n} times is: {avg_time} s")
if len(sys.argv) > 1:
    device = sys.argv[1]
    operator = sys.argv[2]
    version = sys.argv[3]
    epoth = int(sys.argv[4])
    process_distribution(device, operator, version, epoth)
