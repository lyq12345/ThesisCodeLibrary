import cv2
import requests
from picamera2 import Picamera2
import threading
from flask import Flask, Response, render_template
import time
import redis

app = Flask(__name__)

redis_conn = redis.Redis(host='127.0.0.1', port=6379, password='DSM12345', db=0, decode_responses=True)

# camera index
camera_index = 0
# frame data
frame_data = None

# human detection url
# processing_endpoint = 'http://node6:8848/process_video'
# processing_endpoint = 'http://localhost:8848/process_video'

#fire detection url
# processing_endpoint = 'http://node6:8849/process_video'
#processing_endpoint = 'http://localhost:8849/process_video'

def camera_setup():
    def camera_capture():
        width = 4624
        height = 3472

        picam2 = Picamera2()
        camera_config = picam2.create_still_configuration(main={"size": (width, height), "format": "RGB888"},
                                                          buffer_count=1)
        picam2.configure(camera_config)
        picam2.start()
        start_time = time.time()
        while True:
            frame = picam2.capture_array(name="main")
            if frame is None:
                print("None image captured by camera")
                break

            # calculate fps
            current_time = time.time()
            elapsed_time = current_time - start_time
            fps = 1 / elapsed_time
            start_time = current_time

            fps_text = "FPS: {:.2f}".format(fps)

            new_width = 1024
            resized_img = resize_img(frame, new_width)

            # lower the quality
            quality = 65  # quality set to 65%
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

            _, jpeg_frame = cv2.imencode('.jpg', resized_img, encode_params)

            # get successors from redis
            endpoint_cache = redis_conn.get('host1')
            processing_endpoint = str(endpoint_cache)

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

            # put tags on unencoded frame
            cv2.putText(resized_img, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            global frame_data
            frame_data = resized_img

        picam2.stop()

    thread = threading.Thread(target=camera_capture)
    thread.start()

def resize_img(img, new_width):
    aspect_ratio = float(new_width) / img.shape[1]
    new_height = int(img.shape[0] * aspect_ratio)
    resized_image = cv2.resize(img, (new_width, new_height))

    # 去除元信息（strip metadata）
    # 如果图像包含元信息，可以使用Exif标记来删除它
    # 如果没有元信息，可以跳过这一步
    # 示例代码：
    # resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    return resized_image

@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")

def generate_frames():
    while True:
        # cv2.putText(frame, process_time_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        _, jpeg_frame = cv2.imencode('.jpg', frame_data)
        # generate the video stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg_frame.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    camera_setup()
    app.run(host='0.0.0.0', port=5000)

