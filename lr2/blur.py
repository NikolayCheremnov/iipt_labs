import numpy as np
from jupyterlab.utils import deprecated


@deprecated
class Kernels:

    SIMPLE_KERNEL = [
        [1, 1, 1],
        [1, 2, 1],
        [1, 1, 1]
    ]

    MIDDLE_KERNEL = [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 2, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ]

    BIG_KERNEL = [
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
    ]

    LAPLACIAN_KERNEL_1 = [
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0],
    ]

    LAPLACIAN_KERNEL_2 = [
        [1, 1, 1],
        [1, -8, 1],
        [1, 1, 1],
    ]

    @staticmethod
    def get_denominator(kernel):
        denominator = 0
        for row in kernel:
            for item in row:
                denominator += item
        return denominator


@deprecated
def box_blur(src: np.ndarray, kernel=Kernels.SIMPLE_KERNEL):

    src_height = src.shape[0]
    src_width = src.shape[1]

    part = int(len(kernel) / 2)
    denominator = Kernels.get_denominator(kernel)

    dst = np.zeros((src_height, src_width, 3), np.uint8)
    for i in range(src_height):
        for j in range(src_width):
            bgr = [0, 0, 0]
            for k in range(-part, part + 1, 1):
                for p in range(-part, part + 1, 1):
                    if src_height > i + k >= 0 and src_width > j + p >= 0:
                        k_, p_ = k, p
                    else:
                        if i + k >= src_height or i + k < 0:
                            k_ = -k
                        else:
                            k_ = k
                        if j + p >= src_width or j + p < 0:
                            p_ = -p
                        else:
                            p_ = p

                    for ch in range(3):
                        bgr[ch] += src[i + k_][j + p_][ch] * kernel[k + part][p + part]

            for ch in range(3):
                value = bgr[ch] / denominator
                if value > 255:
                    value = 255
                elif value < 0:
                    value = 0
                dst[i][j][ch] = value
    return dst


@deprecated
def get_gaussian_kernel(sigma:float = 1.5, radius: int = 1):
    """
        Generating gaussian kernel
        :param sigma:
        :param radius:
        :return: kernel
    """

    size = radius * 2 + 1
    kernel = np.zeros((size, size), np.float64)

    for i in range(size):
        for j in range(size):
            x, y = i - radius, j - radius
            # TODO: выполнить нормирование
            kernel[i][j] = 1 / (2 * np.pi * sigma ** 2) * np.power(np.e, -(x ** 2 + y ** 2) / (2 * sigma ** 2))

    return kernel


@deprecated
def gaussian_blur(src: np.ndarray, sigma: float = 1.5, radius: int = 1):
    kernel = get_gaussian_kernel(sigma, radius)

    src_height = src.shape[0]
    src_width = src.shape[1]
    part = radius

    dst = np.zeros((src_height, src_width, 3), np.uint8)
    for i in range(src_height):
        for j in range(src_width):
            bgr = [0, 0, 0]
            for k in range(-part, part + 1, 1):
                for p in range(-part, part + 1, 1):
                    if src_height > i + k >= 0 and src_width > j + p >= 0:
                        k_, p_ = k, p
                    else:
                        if i + k >= src_height or i + k < 0:
                            k_ = -k
                        else:
                            k_ = k
                        if j + p >= src_width or j + p < 0:
                            p_ = -p
                        else:
                            p_ = p

                    for ch in range(3):
                        bgr[ch] += src[i + k_][j + p_][ch] * kernel[k + part][p + part]

            for ch in range(3):
                value = bgr[ch]
                if value > 255:
                    value = 255
                elif value < 0:
                    value = 0
                dst[i][j][ch] = value
    return dst


@deprecated
def median_blur(src: np.ndarray, radius: int = 1):
    src_height = src.shape[0]
    src_width = src.shape[1]
    part = radius

    dst = np.zeros((src_height, src_width, 3), np.uint8)
    for i in range(src_height):
        for j in range(src_width):
            ch_list = [[], [], []]
            for k in range(-part, part + 1, 1):
                for p in range(-part, part + 1, 1):
                    if src_height > i + k >= 0 and src_width > j + p >= 0:
                        k_, p_ = k, p
                    else:
                        if i + k >= src_height or i + k < 0:
                            k_ = -k
                        else:
                            k_ = k
                        if j + p >= src_width or j + p < 0:
                            p_ = -p
                        else:
                            p_ = p

                    for ch in range(3):
                        ch_list[ch].append(src[i + k_][j + p_][ch])
            
            # sort lists and set median values
            for ch in range(3):
                ch_list[ch] = sorted(ch_list[ch])
                dst[i][j][ch] = ch_list[ch][int(len(ch_list[ch]) / 2) + 1]

    return dst


@deprecated
def nabla2_laplacian(kernel):
    return (kernel[2, 1] + kernel[0, 1] + kernel[1, 2] + kernel[1, 0]) - 4 * kernel[1][1]


@deprecated
def laplacian_sign(kernel):
    return -1 if kernel[1][1] < 0 else 1


@deprecated
def laplacian_sharpening(src: np.ndarray, kernel=Kernels.LAPLACIAN_KERNEL_2):
    src_height = src.shape[0]
    src_width = src.shape[1]
    coefficient = laplacian_sign(kernel) * laplacian_sign(kernel)

    dst = np.zeros((src_height, src_width, 3), np.uint8)
    for i in range(src_height):
        for j in range(src_width):
            for ch in range(3):
                value = src[i][j][ch] + coefficient
                if value > 255:
                    value = 255
                if value < 0:
                    value = 0
                dst[i][j][ch] = value

    return dst


@deprecated
def get_LoG_kernel(sigma: float = 1.5, radius: int = 1):
    """
        Generating gaussian kernel
        :param sigma:
        :param radius:
        :return: kernel
    """

    size = radius * 2 + 1
    kernel = np.zeros((size, size), np.float64)

    for i in range(size):
        for j in range(size):
            x, y = i - radius, j - radius
            # TODO: нормировать матрицу
            kernel[i][j] = (x ** 2 + y ** 2 - 2 * sigma ** 2) / (sigma ** 4) * np.power(np.e, -(x ** 2 + y ** 2) / (2 * sigma ** 2))
            # kernel[i][j] = -1 / (2 * np.pi * sigma ** 4) * \
            #                (1 - (x ** 2 + y ** 2) / (2 * sigma ** 2)) * \
            #                np.power(np.e, -(x ** 2 + y ** 2) / (2 * sigma ** 2))

    return kernel


@deprecated
def LoG(src: np.ndarray, sigma: float = 1.5, radius: int = 1):
    kernel = [
        [0,1,1,2,2,2,1,1,0],
        [1,2,4,5,5,5,4,2,1],
        [1,4,5,3,0,3,5,4,1],
        [2,5,3,-12,-24,-12,3,5,2],
        [2,5,0,-24,-40,-24,0,5,2],
        [2,5,3,-12,-24,-12,3,5,2],
        [1,4,5,3,0,3,5,4,1],
        [1,2,4,5,5,5,4,2,1],
        [0,1,1,2,2,2,1,1,0],
    ]# get_LoG_kernel(sigma, radius)

    src_height = src.shape[0]
    src_width = src.shape[1]
    part = radius

    dst = np.zeros((src_height, src_width, 3), np.uint8)
    for i in range(src_height):
        for j in range(src_width):
            bgr = [0, 0, 0]
            for k in range(-part, part + 1, 1):
                for p in range(-part, part + 1, 1):
                    if src_height > i + k >= 0 and src_width > j + p >= 0:
                        k_, p_ = k, p
                    else:
                        if i + k >= src_height or i + k < 0:
                            k_ = -k
                        else:
                            k_ = k
                        if j + p >= src_width or j + p < 0:
                            p_ = -p
                        else:
                            p_ = p

                    for ch in range(3):
                        bgr[ch] += src[i + k_][j + p_][ch] * kernel[k + part][p + part]

            for ch in range(3):
                value = bgr[ch]
                if value > 255:
                    value = 255
                elif value < 0:
                    value = 0
                dst[i][j][ch] = value
    return dst
