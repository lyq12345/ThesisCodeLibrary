from flask import Flask, request, render_template, Response
import cv2
import numpy as np
from img_process import detect_human_from_img
import json

app = Flask(__name__)

@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")

@app.route('/process_video', methods=['POST'])
def process_video():
    # 接收视频帧数据
    video_frame = request.data

    # 将图像数据转换为 NumPy 数组
    nparr = np.frombuffer(video_frame, np.uint8)
    # print(nparr)

    # 解码图像数组
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    processed_arr, detections = detect_human_from_img(image)
    detection_json = json.dumps(detections)
    print(detections)

    _, processed_frame = cv2.imencode('.jpg', processed_arr)
    video_stream = processed_frame.tobytes()

    return render_template('index.html', processed_frame=video_stream)



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8848)

