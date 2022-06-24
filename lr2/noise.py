import cv2 as cv
import random
import numpy as np


def sp_noise(image: np.ndarray, probability: float = 0.1):
    """
        Generate noise on image
        :param image: source image
        :param probability: noise probability weight
        :return: new generated image
    """
    output = np.zeros(image.shape, np.uint8)
    thresh = 1 - probability

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < probability:
                output[i][j] = [0, 0, 0]
            elif rdn > thresh:
                output[i][j] = [255, 255, 255]
            else:
                output[i][j] = image[i][j]

    return output


