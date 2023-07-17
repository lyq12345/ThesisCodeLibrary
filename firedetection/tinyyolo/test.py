from imageai.Detection.Custom import CustomVideoObjectDetection
import os

execution_path = os.getcwd()

video_detector = CustomVideoObjectDetection()
video_detector.setModelTypeAsTinyYOLOv3()
video_detector.setModelPath("firedetection/tinyyolo/models/tiny-yolov3_fire-dataset_mAP-0.35102_epoch-71.pt")
video_detector.setJsonPath("firedetection/tinyyolo/models/fire-dataset_tiny-yolov3_detection_config.json")
video_detector.loadModel()

video_detector.detectObjectsFromVideo(input_file_path="firedetection/test_input/DJI_Small_Burn.MP4",
                                        output_file_path=os.path.join(execution_path, "tinyyolo-detected"),
                                        frames_per_second=30,
                                        minimum_percentage_probability=10,
                                        log_progress=True)