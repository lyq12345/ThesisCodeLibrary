import cv2
import requests
from flask import Flask, Response, render_template
import time
import redis
import threading

app = Flask(__name__)


redis_conn = redis.Redis(host='127.0.0.1', port=6379, password='123456', db=0, decode_responses=True)

# camera index
camera_index = 0
frame_data = None

# human detection url
# processing_endpoint = 'http://node6:8848/process_video'
# processing_endpoint = 'http://localhost:8848/process_video'

#fire detection url
# processing_endpoint = 'http://172.22.135.73:8849/process_video'
#processing_endpoint = 'http://localhost:8849/process_video'

def camera_setup():
    def camera_capture():
        capture = cv2.VideoCapture(camera_index)
        start_time = time.time()
        while True:
            ret, frame = capture.read()
            if not ret:
                break

            # 计算帧率
            current_time = time.time()
            elapsed_time = current_time - start_time
            fps = 1 / elapsed_time
            start_time = current_time

            fps_text = "FPS: {:.2f}".format(fps)
            # process_time_text = "Process time: {:.2f} s".format(elapsed_time)
            # encode the captured frame
            _, jpeg_frame = cv2.imencode('.jpg', frame)

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
            cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # cv2.putText(frame, process_time_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            global frame_data
            frame_data = frame

        capture.release()

    thread = threading.Thread(target=camera_capture)
    thread.start()

@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")

def generate_frames():
    while True:
        if frame_data is not None:
            # 将帧数据编码为JPEG格式
            # 计算帧率
            # current_time = time.time()
            # elapsed_time = current_time - start_time
            # fps = 1 / elapsed_time
            # process_time_text = "Process time: {:.2f} s".format(elapsed_time)
            # start_time = current_time
            #
            # # 在帧上绘制帧率
            # fps_text = "FPS: {:.2f}".format(fps)
            # cv2.putText(frame_data, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # cv2.putText(frame_data, process_time_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            _, jpeg = cv2.imencode('.jpg', frame_data)

            # print(fps_text)
            # print(process_time_text)

            # 生成MJPEG流格式数据
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    camera_setup()
    app.run(host='0.0.0.0', port=5000)

