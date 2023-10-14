import cv2
import requests
from flask import Flask, Response, render_template
import time
import redis
import threading

app = Flask(__name__)


# redis_conn = redis.Redis(host='127.0.0.1', port=6379, password='123456', db=0, decode_responses=True)

# camera index
camera_index = 0
frame_data = None

# human detection url
# processing_endpoint = 'http://node6:8848/process_video'
processing_endpoint = 'http://localhost:8848/process_video'

#fire detection url
# processing_endpoint = 'http://172.22.135.73:8849/process_video'
#processing_endpoint = 'http://localhost:8849/process_video'


def camera_setup():
    def camera_capture():
        capture_num = 1
        count = 0
        capture = cv2.VideoCapture(camera_index)
        while True:
            if count >= capture_num:
                break
            ret, frame = capture.read()
            if not ret:
                break

            _, jpeg_frame = cv2.imencode('.jpg', frame)
            print(jpeg_frame.tobytes())

            # try:
            #     response = requests.post(processing_endpoint, data=jpeg_frame.tobytes(),
            #                              headers={'Content-Type': 'image/jpeg'})
            #     response.raise_for_status()
            # except requests.exceptions.Timeout:
            #     print(f'Request to {processing_endpoint} timed out.')
            # except requests.exceptions.HTTPError as err:
            #     print(f'Request to {processing_endpoint} failed with status code {err}')
            # except requests.exceptions.RequestException as e:
            #     print(f'Request to {processing_endpoint} failed')
            #     print(e)
            count += 1

        capture.release()

    thread = threading.Thread(target=camera_capture)
    thread.start()


if __name__ == '__main__':
    camera_setup()
    app.run(host='0.0.0.0', port=5000)

