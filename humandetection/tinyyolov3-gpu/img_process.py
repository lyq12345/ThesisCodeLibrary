from imageai.Detection.Custom import CustomObjectDetection
from imageai.Detection import ObjectDetection
import os
import torch

print(torch.cuda.is_available())

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath(os.path.join(execution_path, "../models/tiny-yolov3.pt"))
detector.loadModel()

def detect_human_from_img(image_arr):
    custom = detector.CustomObjects(person=True)
    output, detections = detector.detectObjectsFromImage(
                            custom_objects=custom,
                            input_image=image_arr,
                            output_type="array",
    minimum_percentage_probability=30)
    for detection in detections:
        print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])
    return output, detections
