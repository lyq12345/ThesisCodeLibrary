from picamera2 import Picamera2
import yaml
import time
import os
import cv2
import requests
# import redis

cur_path = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(cur_path, "capture_config.yml")
f = open(config_path, 'r', encoding='utf-8')
cfg = f.read()
config_dict = yaml.load(cfg, Loader=yaml.FullLoader)
# print(config_dict)

# redis cache
# redis_conn = redis.Redis(host='127.0.0.1', port=6379, password='123456', db=0, decode_responses=True)
processing_endpoint = "http://localhost:8848:process_video"

# camera setup and preview
picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(main={"size": (4656, 3496)}, buffer_count=1)
picam2.configure(camera_config)

raw_path = '/usr/src/raw.jpg'
compressed_path = '/usr/src/compressed.jpg'

def resize_and_encode(img, new_width):
    aspect_ratio = float(new_width) / img.shape[1]
    new_height = int(img.shape[0] * aspect_ratio)
    resized_image = cv2.resize(img, (new_width, new_height))

    # 去除元信息（strip metadata）
    # 如果图像包含元信息，可以使用Exif标记来删除它
    # 如果没有元信息，可以跳过这一步
    # 示例代码：
    # resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    # 设置图像质量
    quality = 65  # 质量为65%
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

    _, jpeg_frame = cv2.imencode('.jpg', resized_image, encode_params)
    return jpeg_frame

try:
    # set capture interval
    capture_interval = 5 if 'interval' not in config_dict else float(config_dict['interval'])  # capture one image every 5 seconds
    picam2.start()
    while True:
        # begin capturing

        # wait for sometime
        time.sleep(capture_interval)

        # # capture one image and save
        # picam2.capture_file(raw_path)
        #
        # # compress the raw image
        # result = subprocess.run(["convert", "-resize", "1024x", "-strip", "-quality", "65%", raw_path, compressed_path], stdout=subprocess.PIPE, text=True)
        # print("image captured!")

        # capture the image and transfer through http
        frame = picam2.capture_array()
        new_width = 1024
        jpeg_frame = resize_and_encode(frame, new_width)

        # send the encoded frame to processors
        try:
            response = requests.post(processing_endpoint, data=jpeg_frame.tobytes(),
                                     headers={'Content-Type': 'image/jpeg'})
            response.raise_for_status()
        except requests.exceptions.Timeout:
            print(f'Request to {processing_endpoint} timed out.')
        except requests.exceptions.HTTPError as err:
            print(f'Request to {processing_endpoint} failed with status code {err}')
        except requests.exceptions.RequestException as e:
            print(f'Request to {processing_endpoint} failed')
            print(e)


except KeyboardInterrupt:
    picam2.stop()
    print("capture stopped")

finally:
    # turn off camera
    picam2.stop()



