import cv2
import numpy as np
import os

execution_path = os.getcwd()

def filter_orange_color(image_arr):
  hsv = cv2.cvtColor(image_arr, cv2.COLOR_BGR2HSV)

  # Threshold of blue in HSV space
  lower_red_orange = np.array([0, 100, 100])
  upper_red_orange = np.array([20, 255, 255])

  # preparing the mask to overlay
  mask = cv2.inRange(hsv, lower_red_orange, upper_red_orange)

  # The black region in the mask has the value of 0,
  # so when multiplied with original image removes all non-blue regions
  processed_image = cv2.bitwise_and(image_arr, image_arr, mask=mask)

  return processed_image

  # cv2_imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))