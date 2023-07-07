from flask import Flask, request
import cv2
import numpy as np
from img_process import detect_human_from_img
import json
import paho.mqtt.client as mqtt

app = Flask(__name__)
mqtt_client = mqtt.Client()
mqtt_client.connect('broker.emqx.io', 1883)
mqtt_client.loop_start()
topic = "iot-1/yuqiaol5/humandetection/bytes"
@app.route('/receive_image', methods=['POST'])
def receive_image():
    image = request.files['image']
    image.save('received_image.jpg')
    return 'Image received and saved successfully'

@app.route('/process_image', methods=['POST'])
def process_image():
    # 获取通过 POST 请求发送的图像数据
    image_data = request.files['image'].read()

    # 将图像数据转换为 NumPy 数组
    nparr = np.frombuffer(image_data, np.uint8)
    # print(nparr)

    # 解码图像数组
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # # 进行图像处理（这里只是一个示例，可以根据需求进行具体的图像处理操作）
    # processed_image = cv2.Canny(image, 100, 200)
    processed_arr, detections = detect_human_from_img(image)
    detection_json = json.dumps(detections)
    mqtt_client.publish(topic, detection_json)
    print(detections)
    # 将处理后的图像编码为 JPEG 格式
    _, processed_jpeg = cv2.imencode('.jpg', processed_arr)

    # 返回处理后的图像数据
    return processed_jpeg.tobytes(), 200, {'Content-Type': 'image/jpeg'}



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8848)

