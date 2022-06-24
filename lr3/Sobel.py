import numpy as np

from lr2.BlurProcessor import BlurProcessor
from lr2.Kernel import Kernel


class Sobel:

    # first mask
    BASE_MASK = [
        [
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ],
        [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ]
    ]


    @staticmethod
    def make_sobel_convolution(src, level=127, masks=BASE_MASK):
        src_height = src.shape[0]
        src_width = src.shape[1]

        dst = np.zeros((src_height, src_width), np.uint8)

        for row in range(src_height):
            for col in range(src_width):
                Gx, Gy = 0, 0
                for k in range(-1, 2, 1):
                    for p in range(-1, 2, 1):
                        x, y = BlurProcessor.convert_kernel_coordinates(k, row, src_height, p, col, src_width)
                        Gx += src[x][y] * masks[0][k + 1][p + 1]
                        Gy += src[x][y] * masks[1][k + 1][p + 1]
                pixel = (Gx ** 2 + Gy ** 2) ** 0.5
                pixel = 0 if pixel < level else 255
                dst[row][col] = pixel

        return dst

    @staticmethod
    def convert_to_gray(src: np.ndarray):
        src_height = src.shape[0]
        src_width = src.shape[1]
        dst = np.zeros((src_height, src_width), np.uint8)

        for row in range(src_height):
            for col in range(src_width):
                dst[row][col] = 0.299 * src[row][col][0] + 0.587 * src[row][col][1] + 0.114 * src[row][col][2]

        return dst
