from ultralytics import YOLO
import os
import cv2

execution_path = os.getcwd()

# model = YOLO("model.pt")

model = YOLO("yolov8n.pt")
# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
results = model.predict(source="0")
# results = model.predict(source="folder", show=True) # Display preds. Accepts all YOLO predict arguments
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
