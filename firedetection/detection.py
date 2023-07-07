from imageai.Detection.Custom import CustomVideoObjectDetection, CustomObjectDetection
from imageai.Detection import VideoObjectDetection, ObjectDetection
import cv2
import os
import time

execution_path = os.getcwd()

inputfolder = "video2images/fire_1m"
# outputfolder = "output"
# inputfolder = "test_input/fire_1m"
outputfolder = "test_output/tiny"

def detect_from_image(inputDir, outputDir):
    detector = CustomObjectDetection()
    # YOLO
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(model_path=os.path.join(execution_path, 'models/yolov3_fire-dataset_last.pt'))
    detector.setJsonPath(configuration_json=os.path.join(execution_path, "models/fire-dataset_yolov3_detection_config.json"))

    # TinyYOLO
    # detector.setModelTypeAsTinyYOLOv3()
    # detector.setModelPath(model_path=os.path.join(execution_path, 'models/tiny-yolov3_fire-dataset_mAP-0.35102_epoch-71.pt'))
    # detector.setJsonPath(configuration_json=os.path.join(execution_path, "models/fire-dataset_tiny-yolov3_detection_config.json"))


    detector.loadModel()

    extensions = ('.jpg', '.jpeg', '.png', '.gif')

    files = [f for f in os.listdir(inputDir) if not f.startswith('.DS_Store')]
    distance_name = inputDir.split('/')[-1]
    outputDir = os.path.join(outputDir, distance_name)
    detection_count = 0
    positive_count = 0
    process_time_sum = 0.0
    confidence_sum = 0.0

    accuracy = 0.0
    avg_confidence = 0.0
    avg_time = 0.0

    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    for filename in files:
        file_name, file_extension = os.path.splitext(filename)
        if file_extension not in extensions:
            continue
        image_path = os.path.join(inputDir, filename)
        print("Processing image: " + filename)
        start_time = time.time()
        returned_image, detection = detector.detectObjectsFromImage(input_image=image_path,
                                                                    display_percentage_probability=False,
                                                                    display_object_name=False,
                                                     output_type="array",
                                                     minimum_percentage_probability=20,
                                                                    objectness_treshold=0.3
                                                                    )
        end_time = time.time()
        run_time = end_time - start_time
        process_time_sum += run_time

        for eachObject in detection:
            probability = eachObject["percentage_probability"]
            probability = float(probability)
            confidence_sum += probability
            positive_count += 1

        if(len(detection) != 0):
            cv2.imwrite(os.path.join(outputDir, "detected_"+filename), returned_image)
            detection_count += 1
        else:
            cv2.imwrite(os.path.join(outputDir, filename), returned_image)

    avg_time = round((process_time_sum / 300), 4)
    print("Average process time: " + str(avg_time))

    if positive_count != 0:
        accuracy = round((detection_count / 300), 4)
        avg_confidence = round((confidence_sum / positive_count), 4)
        print("Average confidence: " + str(avg_confidence))

        print("Accuracy: " + str(accuracy))
    else:
        print("No detection result")

    return accuracy, avg_time, avg_confidence

def detect_human_from_image():
    # https://github.com/miquelmarti/Okutama-Action
    # https: // github.com / accenture / air
    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(path="retinanet_resnet50_fpn_coco-eeacb38b.pth")
    detector.loadModel()
    custom = detector.CustomObjects(person=True)

    extensions = ('.jpg', '.jpeg', '.png', '.gif')

    files = [f for f in os.listdir(inputfolder) if not f.startswith('.DS_Store')]
    for filename in files:
        file_name, file_extension = os.path.splitext(filename)
        if file_extension not in extensions:
            continue
        image_path = os.path.join(inputfolder, filename)
        print("Processing image: " + file_name)
        detections = detector.detectObjectsFromImage(custom_objects=custom, input_image=image_path,
                                                     output_image_path=os.path.join(outputfolder, "result_" + filename),
                                                     minimum_percentage_probability=40)
def detect_from_video():
    video_detector = CustomVideoObjectDetection()
    video_detector.setModelTypeAsYOLOv3()
    # video_detector.setModelPath("yolov3.pt")
    # video_detector.setModelTypeAsRetinaNet()
    video_detector.setModelPath(model_path="models/yolov3_fire-dataset_last.pt")
    video_detector.setJsonPath("fire-dataset_yolov3_detection_config.json")
    video_detector.loadModel()

    # custom = video_detector.CustomObjects(person=True)

    video_detector.detectObjectsFromVideo(
        # input_file_path="video/fire_12m.mp4",
        input_file_path="fire_cut.mp4",
                                          output_file_path=os.path.join(execution_path, "fire_cut_detected"),
                                          frames_per_second=30,
                                          minimum_percentage_probability=40,
                                          log_progress=True)

def detect_human_from_video():
    video_detector = VideoObjectDetection()
    video_detector.setModelTypeAsRetinaNet()
    video_detector.setModelPath("retinanet_resnet50_fpn_coco-eeacb38b.pth")
    # video_detector.setModelPath(model_path="yolov3_fire-dataset_last.pt")
    # video_detector.setJsonPath("fire-dataset_yolov3_detection_config.json")
    video_detector.loadModel()

    custom = video_detector.CustomObjects(person=True)

    video_detector.detectObjectsFromVideo(
        input_file_path="human_cut.mp4",
        # input_file_path="fire_cut.mp4",
        custom_objects=custom,
                                          output_file_path=os.path.join(execution_path, "human_cut_detected"),
                                          frames_per_second=30,
                                          minimum_percentage_probability=40,
                                          log_progress=True)

# detect_from_image(inputfolder, outputfolder)
# detect_human_from_image()
# detect_from_video()
# detect_human_from_video()