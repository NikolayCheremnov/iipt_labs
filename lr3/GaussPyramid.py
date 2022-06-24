from functools import reduce
from typing import List

import numpy as np

from lr2.BlurProcessor import BlurProcessor
from lr2.Kernel import Kernel


class GaussPyramid:

    @staticmethod
    def make_next_layer(src: np.ndarray, sigma: float = 0.7, radius: int = 2, use_blur = True):
        if use_blur:
            blur_img = BlurProcessor.make_convolution(src, Kernel.get_gaussian_kernel(sigma, radius))
        else:
            blur_img = src

        src_height = src.shape[0]
        src_width = src.shape[1]
        dst = np.zeros((src_height // 2, src_width // 2, 3), np.float64)

        new_i = 0
        for i in range(0, src_height, 2):
            new_j = 0
            for j in range(0, src_width, 2):
                dst[new_i][new_j] = blur_img[i][j]
                new_j += 1
            new_i += 1

        return dst

    @staticmethod
    def draw_pyramids(images: List[np.ndarray]):
        images.sort(key=lambda x: x.shape[1], reverse=True)
        canvas_height = images[0].shape[0]
        canvas_width = reduce(lambda res, image: res + image.shape[1], images, 0)

        canvas = np.zeros((canvas_height, canvas_width, 3), np.float64)

        offset = 0

        for img in images:
            for row in range(img.shape[0]):
                for col in range(img.shape[1]):
                    canvas[row][col + offset] = img[row][col]
            offset += img.shape[1]

        return canvas

