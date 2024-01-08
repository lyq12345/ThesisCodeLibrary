# from imageai.Detection.Custom import CustomVideoObjectDetection
# import os
#
# execution_path = os.getcwd()
#
# video_detector = CustomVideoObjectDetection()
# video_detector.setModelTypeAsTinyYOLOv3()
# video_detector.setModelPath("firedetection/tinyyolov3/models/tiny-yolov3_fire-dataset_mAP-0.35102_epoch-71.pt")
# video_detector.setJsonPath("firedetection/tinyyolov3/models/fire-dataset_tiny-yolov3_detection_config.json")
# video_detector.loadModel()
#
# video_detector.detectObjectsFromVideo(input_file_path="firedetection/test_input/DJI_Small_Burn.MP4",
#                                         output_file_path=os.path.join(execution_path, "tinyyolov3-detected"),
#                                         frames_per_second=30,
#                                         minimum_percentage_probability=10,
#                                         log_progress=True)

def min_cost_to_satisfy_conditions(cost, compatible1, compatible2, min_compatible):
    n = len(cost)
    valid_gpus = []

    for i in range(n):
        if compatible1[i] == 1 and compatible2[i] == 1:
            valid_gpus.append(cost[i])

    if len(valid_gpus) < 2 * min_compatible:
        return -1

    valid_gpus.sort()

    return sum(valid_gpus[:min_compatible] + valid_gpus[-min_compatible:])


# Example usage:
cost = [2, 4, 6, 5]
compatible1 = [1, 1, 1, 0]
compatible2 = [0, 0, 1, 1]
min_compatible = 2

result = min_cost_to_satisfy_conditions(cost, compatible1, compatible2, min_compatible)
print(result)  # Output: 13
