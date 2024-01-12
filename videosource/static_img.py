import time
import requests
import cv2

# human detection url
# processing_endpoint = 'http://node6:8848/process_video'
# processing_endpoint = 'http://localhost:8848/process_video'
processing_endpoint = 'http://128.200.218.112:8849/process_video'

#fire detection url
# processing_endpoint = 'http://node6:8849/process_video'
# processing_endpoint = 'http://localhost:8849/process_video'

# 定义发送请求的间隔时间（秒）
interval = 1
# camera_index = 0
# cap = cv2.VideoCapture(camera_index)
img_path = "test_imgs/fire.jpg"
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
    # ret, frame = cap.read()

    # calculate fps
    # current_time = time.time()
    # elapsed_time = current_time - start_time
    # fps = 1 / elapsed_time
    # start_time = current_time
    #
    # fps_text = "FPS: {:.2f}".format(fps)

    # print("1111")

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

    # 等待指定的时间间隔
    # time.sleep(interval)
# cap.release()