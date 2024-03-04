import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import numpy as np
import cv2

execution_path = os.getcwd()

# STEP 2: Create an PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path=os.path.join(execution_path, '../models/pose_landmarker.task'))
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

def detect_and_draw_landmarks(image_arr):
  image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_arr)
  # print(type(image))

  # STEP 4: Detect pose landmarks from the input image.
  detection_result = detector.detect(image)

  # STEP 5: Process the detection result. In this case, visualize it.
  annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
  return annotated_image
  # annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
  # cv2.imwrite("pose_output.jpg", annotated_image)

# cv2_imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))