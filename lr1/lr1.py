import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys


class Configurations:
    DATA_PATH = './data/'
    FILE_NAME = 'bears.jpg'

    @staticmethod
    def get_full_name():
        return Configurations.DATA_PATH + Configurations.FILE_NAME

    @staticmethod
    def get_modified_name(modifier: str):
        return Configurations.DATA_PATH + modifier + '_' + Configurations.FILE_NAME
    
    # blur configs
    KERNEL_SIZE = (11, 11)
    SIGMA_X = 0

    # contouring configs
    HSV_MIN = np.array((2, 28, 65), np.uint8)
    HSV_MAX = np.array((26, 238, 255), np.uint8)

    # segmentation ranges
    BGR_MIN = np.array((47, 47, 82), np.uint8)
    BGR_MAX = np.array((220, 230, 255), np.uint8)


def run():
    # reading source image
    img = cv.imread(Configurations.get_full_name())
    if img is None:
        sys.exit('Could not read the image.')
    # cv.imshow('Source image', img)
    # k = cv.waitKey(0)

    """
        Blur:
            - averaging
            - gaussian
            - median
    """
    # 1. averaging blur
    img_averaging_blur = cv.blur(img, Configurations.KERNEL_SIZE)
    # cv.imshow(f'Blur {Configurations.KERNEL_SIZE}', img_averaging_blur )
    # k = cv.waitKey(0)
    cv.imwrite(Configurations.get_modified_name(
        f'averaging_blur_{Configurations.KERNEL_SIZE}'), img_averaging_blur)

    # 2. gaussian blur
    img_gaussian_blur = cv.GaussianBlur(img, Configurations.KERNEL_SIZE, Configurations.SIGMA_X)
    cv.imwrite(Configurations.get_modified_name(
        f'gaussian_blur_{Configurations.KERNEL_SIZE}_{Configurations.SIGMA_X}'), img_gaussian_blur)

    # 3. median blue
    img_median_blur = cv.medianBlur(img, Configurations.KERNEL_SIZE[0])
    cv.imwrite(Configurations.get_modified_name(
        f'median_blur_{Configurations.KERNEL_SIZE[0]}'), img_median_blur)

    """
        Contours selection
    """
    # 1. preparing image
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    thresh = cv.inRange(hsv, Configurations.HSV_MIN, Configurations.HSV_MAX)

    # 2. find contours
    contours, hierarchy = cv.findContours(thresh.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # 3. add contours to image
    contoured_img = img.copy()
    cv.drawContours(
        image=contoured_img,
        contours=contours,
        contourIdx=-1,
        color=(0, 0, 255),
        lineType=cv.LINE_AA,
        hierarchy=hierarchy,
        thickness=3,
        maxLevel=1)
    cv.imwrite(Configurations.get_modified_name('contoured'), contoured_img)

    """
        BGR segmentation
    """
    mask = cv.inRange(img, Configurations.BGR_MIN, Configurations.BGR_MAX)
    segmented = cv.bitwise_and(img, img, mask=mask)
    cv.imwrite(Configurations.get_modified_name('segm'), segmented)

    """
        BGR segmentation and Dice
    """
    img_t = cv.cvtColor(cv.imread(Configurations.get_full_name()), cv.COLOR_BGR2RGB)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img_t, contours, -1, (0, 0, 255), -1)
    cv.imwrite(Configurations.get_modified_name('imt_t'), img_t)
    
    img_cont = cv.cvtColor(cv.imread(Configurations.get_modified_name('cont')), cv.COLOR_BGR2RGB)

    # finding red and white pixels
    counter, red_counter, white_counter = 0, 0, 0

    for row in range(img_t.shape[0]):
        for col in range(img_t.shape[1]):
            is_red, is_white = False, False
            # check red pixel
            if img_t[row, col, 0] == 0 and img_t[row, col, 1] == 0 and img_t[row, col, 2] == 255:
                is_red = True
            # check white pixel
            if img_cont[row, col, 0] == 255 and img_cont[row, col, 1] == 255 and img_cont[row, col, 2] == 255:
                is_white = True
            if is_white and is_red:
                counter += 1
            if is_red:
                red_counter += 1
            if is_white:
                white_counter += 1

    print(100 * 2 * counter / (red_counter + white_counter))
