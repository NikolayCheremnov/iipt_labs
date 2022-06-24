import numpy as np


class BlurProcessor:

    @staticmethod
    def make_convolution(src: np.ndarray, kernel: np.ndarray):
        src_height = src.shape[0]
        src_width = src.shape[1]

        part = len(kernel) // 2
        dst = np.zeros((src_height, src_width, 3), np.float64)

        for row in range(src_height):
            for col in range(src_width):
                pixel = np.array([0, 0, 0], np.float64)
                for k in range(-part, part + 1, 1):
                    for p in range(-part, part + 1, 1):
                        x, y = BlurProcessor.convert_kernel_coordinates(k, row, src_height, p, col, src_width)
                        pixel += src[x][y] * kernel[k + part][p + part]

                pixel[0] = 0 if pixel[0] < 0 else 255 if pixel[0] > 255 else pixel[0]
                pixel[1] = 0 if pixel[1] < 0 else 255 if pixel[1] > 255 else pixel[1]
                pixel[2] = 0 if pixel[2] < 0 else 255 if pixel[2] > 255 else pixel[2]

                dst[row][col] = pixel
        return dst

    @staticmethod
    def make_median_convolution(src: np.ndarray, radius: int = 2):
        src_height = src.shape[0]
        src_width = src.shape[1]
        part = radius

        dst = np.zeros((src_height, src_width, 3), np.uint8)
        for row in range(src_height):
            for col in range(src_width):
                ch_list = [[], [], []]
                for k in range(-part, part + 1, 1):
                    for p in range(-part, part + 1, 1):
                        x, y = BlurProcessor.convert_kernel_coordinates(k, row, src_height, p, col, src_width)
                        ch_list[0].append(src[x][y][0])
                        ch_list[1].append(src[x][y][1])
                        ch_list[2].append(src[x][y][2])

                # sort lists and set median values
                for ch in range(3):
                    ch_list[ch] = sorted(ch_list[ch])
                    dst[row][col][ch] = ch_list[ch][int(len(ch_list[ch]) / 2) + 1]

        return dst

    @staticmethod
    def convert_kernel_coordinates(k: int, row: int, src_height: int, p: int, col: int, src_width: int):
        return max(0, min(row + k, src_height - 1)), \
               max(0, min(col + p, src_width - 1))
