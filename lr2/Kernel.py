import numpy as np


class Kernel:
    SIMPLE_KERNEL = np.array([
        [1, 1, 1],
        [1, 2, 1],
        [1, 1, 1]
    ])

    MIDDLE_KERNEL = np.array([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 2, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ], np.int32)

    BIG_KERNEL = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ])

    @staticmethod
    def make_kernel_normalization(kernel):
        return kernel * (1 / np.sum(kernel))

    @staticmethod
    def get_gaussian_kernel(sigma: float = 1.5, radius: int = 1):
        size = radius * 2 + 1
        kernel = np.zeros((size, size), np.float64)

        for i in range(size):
            for j in range(size):
                x, y = i - radius, j - radius
                kernel[i][j] = 1 / (2 * np.pi * sigma ** 2) * np.power(np.e, -(x ** 2 + y ** 2) / (2 * sigma ** 2))

        return Kernel.make_kernel_normalization(kernel)

    @staticmethod
    def create_log_kernel(sigma: float = 1.5, radius: int = 3):
        size = radius * 2 + 1
        kernel = np.zeros((size, size), np.float64)

        for i in range(size):
            for j in range(size):
                x, y = i - radius, j - radius
                kernel[i][j] = (x ** 2 + y ** 2 - 2 * sigma ** 2) / (sigma ** 4) * \
                               np.power(np.e, -(x ** 2 + y ** 2) / (2 * sigma ** 2))
        return Kernel.make_kernel_normalization(kernel)
