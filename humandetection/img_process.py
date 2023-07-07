from imageai.Detection.Custom import CustomObjectDetection
from imageai.Detection import ObjectDetection

detector1 = ObjectDetection()
detector1.setModelTypeAsYOLOv3()
detector1.setModelPath("models/yolov3.pt")
detector1.loadModel()

def detect_human_from_img(image_arr):
    custom = detector1.CustomObjects(person=True)
    output, detections = detector1.detectObjectsFromImage(
                            custom_objects=custom,
                            input_image=image_arr,
                            output_type="array",
    minimum_percentage_probability=30)
    for detection in detections:
        print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])
    return output, detections
