'''
This file houses all relevant image processing
functions
'''

'''
NOTE: THIS IMAGE PROCESSING ONLY WORKED IN OUR SIMULATION
OTHER COLORS AND CAMERAS NOT TESTED; MAY NOT WORK FOR CAMERAS
THAT TAKE BRIGHTER IMAGES OR SIMULATIONS WITH BRIGHTER TERRAIN
'''
import cv2
import numpy as np
# from glob import glob
from skimage import measure
import imutils
from imutils import contours
import os


execution_path = os.getcwd()

def crop(path: str, height_start: int, height_end: int, width_start: int, width_end: int):
    '''
    Crops the image given by the path according to the parameters
    '''
    cv2.imwrite(path, cv2.imread(path)[height_start:height_end, width_start: width_end])


def detect_fire(image):

    # Seperates the fire areas from everything else
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 30, 240])
    upper_red = np.array([255, 255, 255])

    mask = cv2.inRange(hsv, lower_red, upper_red)
    fires = cv2.bitwise_and(image, image, mask=mask)

    # Converts into grayscale
    gray = cv2.cvtColor(fires, cv2.COLOR_BGR2GRAY)

    # Differentiates the fire areas and the background areas
    thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1]

    # perform a connected component analysis on the thresholded
    # image, then initialize a mask to store only the "large"
    # components
    labels = measure.label(thresh, background=0)
    mask = np.zeros(thresh.shape, dtype="uint8")

    # loop over the unique components
    for label in np.unique(labels):

        # if this is the background label, ignore it
        if label == 0:
            continue

        # otherwise, construct the label mask and count the
        # number of pixels
        labelMask = np.zeros(thresh.shape, dtype="uint8")

        labelMask[labels == label] = 255

        numPixels = cv2.countNonZero(labelMask)

        # if the number of pixels in the component is sufficiently
        # large, then add it to our mask of "large blobs"
        if numPixels > 600:
            mask = cv2.add(mask, labelMask)

    # find the contours in the mask, then sort them from left to
    # right
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # list that will be returned containing data
    fire_list = []

    if len(cnts) > 0:
        cnts = contours.sort_contours(cnts)[0]

        # loop over the contours
        for (_, c) in enumerate(cnts):
            # draw the bright spot on the image
            (x, y, w, h) = cv2.boundingRect(c)

            fire_list.append((x, y, w, h))

            cv2.rectangle(image, (x, y), (x + w, y + h), (153, 0, 76), 5)

    # cv2.imwrite(output, image)
    return image, fire_list


def calc_location_fire(location, fire_list: int = []):
    '''
    calculates the size and location of any potential fire
    in a processed image given the height of the drone
    '''

    # We first calculate the length and width of the area captured by the drone

    # we must adjust the calculation with respect to the mirror
    mirror_constant = 0.5
    diagonal_length = 2 * 0.87852146605 * location[2] * mirror_constant  # 0.87852146605 = cos(41.3 degrees)
    width = diagonal_length * 0.8
    length = diagonal_length * 0.6

    print(length, width)

    # Then we calculate where the fires are
    # conversion between pixel and cm
    px_to_cm = width / 960.0

    # Saves our mapped fires
    mapped_fires = []

    for fire in fire_list:
        fx, fy, fw, fh = fire

        fx, fy = (fx - 480) * px_to_cm + location[0], (360 - fy) * px_to_cm + location[1]
        fw, fh = fw * px_to_cm, fh * px_to_cm

        mapped_fires.append({"fx": int(fx), "fy": int(fy), "fw": int(fw), "fh": int(fh)})

    return mapped_fires


# for testing
# if __name__ == "__main__":
#
#     i = 0
#
#     for path in glob("./test_images/*"):
#
#         # detect_grid(path, f'./test_images_results/image_waypoint_{i}_results.png')
#
#         # detect_fire(f'./test_images_results/image_waypoint_{i}_results.png', f'./test_images_results/image_waypoiny_{i}_results.png')
#         image = cv2.imread(path)
#         output, fire_list = detect_fire(image)
#
#         print(fire_list, i)
#         # cv2.imwrite("result.png", output)
#         # fires = calc_location_fire((100, 100, 100), fire_list)
#         # print(fires)
#
#         # with open(f'fire_data_{i}.json', 'w') as f:
#         #     json.dump(fires, f)
#
#         i += 1