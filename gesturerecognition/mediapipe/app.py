# -*- coding: UTF-8 -*-
from flask import Flask, request, render_template, Response
import cv2
import numpy as np
from img_process import detect_and_draw_landmarks
import json
import time

app = Flask(__name__)

frame_data = None
camera_index = 0

def generate_frames():
    start_time = time.time()
    while True:
        if frame_data is not None:
            ## encode into jpeg format
            ## compute fps
            # current_time = time.time()
            # elapsed_time = current_time - start_time
            # fps = 1 / elapsed_time
            # process_time_text = "Process time: {:.2f} s".format(elapsed_time)
            # start_time = current_time
            #
            # # draw fps text
            # fps_text = "FPS: {:.2f}".format(fps)
            # cv2.putText(frame_data, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # cv2.putText(frame_data, process_time_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            _, jpeg = cv2.imencode('.jpg', frame_data)

            # print(fps_text)
            # print(process_time_text)

            # generate jpeg bytestream
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
    start_time = time.time()
    # receive frame data from HTTP
    video_frame = request.data
    result = {}

    # transfer the byte data to nparr
    nparr = np.frombuffer(video_frame, np.uint8)

    # decode
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    processed_arr = detect_and_draw_landmarks(image)
    frame_data = processed_arr

    end_time = time.time()
    elapsed_time = end_time - start_time
    process_time_text = "Process time: {:.2f} s".format(elapsed_time)

    print(process_time_text)

    result['detections'] = []
    result['process_time'] = elapsed_time
    result_json = json.dumps(result)

    print(result_json)

    return result_json



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8850)

