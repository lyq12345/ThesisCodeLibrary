from flask import Flask, request, render_template, Response
import cv2
import numpy as np
from img_process import detect_fire_from_img
import json
import time

app = Flask(__name__)

frame_data = None
camera_index = 0

def generate_frames():
    while True:
        if frame_data is not None:
            # 将帧数据编码为JPEG格式
            _, jpeg = cv2.imencode('.jpg', frame_data)

            # 生成MJPEG流格式数据
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/process_video', methods=['POST'])
def process_video():
    global frame_data
    # 接收视频帧数据
    video_frame = request.data

    # transfer the byte data to nparr
    nparr = np.frombuffer(video_frame, np.uint8)

    # decode
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    processed_arr, detections = detect_fire_from_img(image)
    frame_data = processed_arr
    detection_json = json.dumps(detections)
    print(detections)

    return detection_json



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8849)

