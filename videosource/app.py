import cv2
import requests
import numpy as np
from flask import Flask, Response, render_template
import time

app = Flask(__name__)

# 摄像头索引
camera_index = 0

# human detection url
# processing_endpoint = 'http://172.22.135.73:8848/process_video'
# processing_endpoint = 'http://localhost:8848/process_video'

#fire detection url
# processing_endpoint = 'http://172.22.135.73:8849/process_video'
processing_endpoint = 'http://localhost:8849/process_video'
@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")

def generate_frames():
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

        # 在帧上绘制帧率
        fps_text = "FPS: {:.2f}".format(fps)
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # encode the captured frame
        _, jpeg_frame = cv2.imencode('.jpg', frame)

        # send the encoded frame to processors
        response = requests.post(processing_endpoint, data=jpeg_frame.tobytes(), headers={'Content-Type': 'image/jpeg'})

        # 处理端的响应（可选）
        print(response.text)

        # 生成视频流
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg_frame.tobytes() + b'\r\n')

    capture.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
