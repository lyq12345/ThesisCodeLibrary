import time
import requests
import cv2
import sys

host_device = sys.argv[1]
port_number = sys.argv[2]
operator_type = sys.argv[3]
interval = float(sys.argv[4])

hosts = {
    'pi': '111',
    'nano': '128.200.218.98',
    'xavier': '128.200.218.123'
}

host_ip = hosts[host_device]
# human detection url
# processing_endpoint = 'http://node6:8848/process_video'
# processing_endpoint = 'http://localhost:8848/process_video'
processing_endpoint = f'http://{host_ip}:{port_number}/process_video'

#fire detection url
# processing_endpoint = 'http://node6:8849/process_video'
# processing_endpoint = 'http://localhost:8849/process_video'

# camera_index = 0
# cap = cv2.VideoCapture(camera_index)
# img_path = "test_imgs/fire.jpg"

img_path = "test_imgs/human.jpeg" if operator_type=="human" else "test_imgs/fire.jpg"
frame = cv2.imread(img_path)

def resize_img(img, new_width):
    aspect_ratio = float(new_width) / img.shape[1]
    new_height = int(img.shape[0] * aspect_ratio)
    resized_image = cv2.resize(img, (new_width, new_height))
    return resized_image

new_width = 1024
resized_img = resize_img(frame, new_width)

# lower the quality
quality = 65  # quality set to 65%
encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

_, jpeg_frame = cv2.imencode('.jpg', resized_img, encode_params)
processing_endpoint = str(processing_endpoint)

while True:
    try:
        response = requests.post(processing_endpoint, data=jpeg_frame.tobytes(),
                                 headers={'Content-Type': 'image/jpeg'})
        if response.status_code == 200:
            result = response.json()
            proc_time = round(float(result['process_time']), 4)
            print(f"{proc_time}")
        response.raise_for_status()
    except requests.exceptions.Timeout:
        print(f'Request to {processing_endpoint} timed out.')
    except requests.exceptions.HTTPError as err:
        print(f'Request to {processing_endpoint} failed with status code {err}')
    except requests.exceptions.RequestException as e:
        print(f'Request to {processing_endpoint} failed')
        print(e)

    # 等待指定的时间间隔
    time.sleep(interval)
# cap.release()