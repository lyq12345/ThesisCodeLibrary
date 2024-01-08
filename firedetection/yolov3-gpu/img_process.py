from imageai.Detection.Custom import CustomObjectDetection
import os

execution_path = os.getcwd()

# YOLOv3
detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(model_path=os.path.join(execution_path, '../models/yolov3_fire-dataset_last.pt'))
detector.setJsonPath(configuration_json=os.path.join(execution_path,
                                                    "../models/fire-dataset_yolov3_detection_config.json"))
detector.loadModel()

def detect_fire_from_img(image_arr):
    output, detections = detector.detectObjectsFromImage(
                            input_image=image_arr,
                            output_type="array",
                            minimum_percentage_probability=20,
                            objectness_treshold=0.3
                            )
    for detection in detections:
        print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])
    return output, detections
