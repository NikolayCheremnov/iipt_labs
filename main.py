import cv2 as cv
import numpy as np

from lr1 import lr1
from lr2 import noise, plot, blur
from lr2.BlurProcessor import BlurProcessor
from lr2.Kernel import Kernel
from lr2.blur import Kernels
from lr3.GaussPyramid import GaussPyramid
from lr3.Sobel import Sobel


def test_lr_2(path, src, run_mask):
    try:
        # 1. read source image
        img = cv.imread(path + src)
        print('> Color image read')

        # 2. create noise
        if run_mask[0]:
            try:
                noise_img_1 = noise.sp_noise(img, 0.01)
                cv.imwrite(f'{path}noise/noise_001.png', noise_img_1)
                noise_img_2 = noise.sp_noise(img, 0.05)
                cv.imwrite(f'{path}noise/noise_005.png', noise_img_2)
                noise_img_3 = noise.sp_noise(img, 0.1)
                cv.imwrite(f'{path}noise/noise_010.png', noise_img_3)
                print('> Noise created')
            except Exception as ex:
                print(f'> Noise corrupted by exception: {ex}')

        # 3. make box blur
        if run_mask[1]:
            try:
                box_blur_img_1 = BlurProcessor.make_convolution(img, Kernel.make_kernel_normalization(Kernel.MIDDLE_KERNEL))
                cv.imwrite(f'{path}box_blur/middle_blur.png', box_blur_img_1)
                box_blur_img_2 = BlurProcessor.make_convolution(img, Kernel.make_kernel_normalization(Kernel.BIG_KERNEL))
                cv.imwrite(f'{path}box_blur/big_blur.png', box_blur_img_2)
                box_blur_img_3 = BlurProcessor.make_convolution(noise_img_1, Kernel.make_kernel_normalization(Kernel.MIDDLE_KERNEL))
                cv.imwrite(f'{path}box_blur/noise_middle_blur.png', box_blur_img_3)
                box_blur_img_4 = BlurProcessor.make_convolution(noise_img_1, Kernel.make_kernel_normalization(Kernel.BIG_KERNEL))
                cv.imwrite(f'{path}box_blur/noise_big_blur.png', box_blur_img_4)
                print('> Box blur processed')
            except Exception as ex:
                print(f'> Box blur corrupted by exception: {ex}')

        # 4. make gaussian blur
        if run_mask[2]:
            try:
                gaussian_images = []
                for processed, info in zip([img, noise_img_1], ['img', 'noise_img']):
                    for sigma in [1.4, 0.7]:
                        for radius in [2, 3, 4]:
                            gaussian_img = BlurProcessor.make_convolution(processed, Kernel.get_gaussian_kernel(sigma, radius))
                            cv.imwrite(f'{path}gaussian_blur/{info}_gaussian_blur_{sigma}_{radius}.png', gaussian_img)
                            gaussian_images.append(gaussian_img)
                print('> Gaussian blur processed')
            except Exception as ex:
                print(f'> Gaussian blur corrupted by exception: {ex}')

        # 5. make median blur
        if run_mask[3]:
            try:
                median_images = []
                for processed, info in zip([img, noise_img_1], ['img', 'noise_img']):
                    for radius in [2, 3, 4]:
                        median_img = BlurProcessor.make_median_convolution(processed, radius)
                        cv.imwrite(f'{path}median_blur/{info}_median_blur_{radius}.png', median_img)
                        median_images.append(median_img)
                print('> Median blur processed')
            except Exception as ex:
                print(f'> Median blur corrupted by exception: {ex}')

        # 6. log filter
        if run_mask[4]:
            try:
                for processed, info in zip([img, noise_img_1], ['img', 'noise_img']):
                    for sigma in [1.4, 0.7]:
                        for radius in [2, 3, 4]:
                            img_log = BlurProcessor.make_convolution(processed, Kernel.create_log_kernel(sigma, radius))
                            cv.imwrite(f'{path}log/{info}_log_filter_{sigma}_{radius}.png', img_log)
                print('> Log filter processed')
            except Exception as ex:
                print(f'> LoG corrupted by exception: {ex}')
                
    except Exception as ex:
        print(f'> Test corrupted by exception: {ex}.')
        print(f'> Args:\n{path}\n{src}\n{run_mask}')


def save_pyramids(path, dst, pyramid_images):
    pyramid_image = GaussPyramid.draw_pyramids(pyramid_images)
    cv.imwrite(path + dst, pyramid_image)
    gray_pyramid = Sobel.convert_to_gray(pyramid_image)
    cv.imwrite(f'{path}gray_{dst}', gray_pyramid)
    sobel_pyramid = Sobel.make_sobel_convolution(gray_pyramid)
    cv.imwrite(f'{path}sobel_{dst}', sobel_pyramid)


def test_lr_3(path: str, src: str):
    try:
        img = cv.imread(path + src)
        print('> image read')

        images = [img]

        # 1. first layer
        layer_1_clear = GaussPyramid.make_next_layer(img, use_blur=False)
        layer_1_0_7 = GaussPyramid.make_next_layer(img, 0.7, 5)
        layer_1_1_5 = GaussPyramid.make_next_layer(img, 1.5, 5)
        print('> layers 1 created')

        # 2. second layer
        layer_2_clear = GaussPyramid.make_next_layer(layer_1_clear, use_blur=False)

        layer_2_0_7_0_7 = GaussPyramid.make_next_layer(layer_1_0_7, 0.7, 5)
        layer_2_0_7_1_5 = GaussPyramid.make_next_layer(layer_1_0_7, 1.5, 5)

        layer_2_1_5_0_7 = GaussPyramid.make_next_layer(layer_1_1_5, 0.7, 5)
        layer_2_1_5_1_5 = GaussPyramid.make_next_layer(layer_1_1_5, 1.5, 5)
        print('> layers 2 created')

        # 3. third layer
        layer_3_clear = GaussPyramid.make_next_layer(layer_2_clear, use_blur=False)

        layer_3_0_7_0_7_0_7 = GaussPyramid.make_next_layer(layer_2_0_7_0_7, 0.7, 5)
        layer_3_0_7_0_7_1_5 = GaussPyramid.make_next_layer(layer_2_0_7_0_7, 1.5, 5)

        layer_3_0_7_1_5_0_7 = GaussPyramid.make_next_layer(layer_2_0_7_1_5, 0.7, 5)
        layer_3_0_7_1_5_1_5 = GaussPyramid.make_next_layer(layer_2_0_7_1_5, 1.5, 5)

        layer_3_1_5_0_7_0_7 = GaussPyramid.make_next_layer(layer_2_1_5_0_7, 0.7, 5)
        layer_3_1_5_0_7_1_5 = GaussPyramid.make_next_layer(layer_2_1_5_0_7, 1.5, 5)

        layer_3_1_5_1_5_0_7 = GaussPyramid.make_next_layer(layer_2_1_5_1_5, 0.7, 5)
        layer_3_1_5_1_5_1_5 = GaussPyramid.make_next_layer(layer_2_1_5_1_5, 1.5, 5)
        print('> layers 3 created')

        # make pyramids lists
        clear_pyramid_images = [img, layer_1_clear, layer_2_clear, layer_3_clear]
        save_pyramids(path, f'clear_{src}', clear_pyramid_images)

        pyramid_0_7_0_7_0_7 = [img, layer_1_0_7, layer_2_0_7_0_7, layer_3_0_7_0_7_0_7]
        save_pyramids(path, f'0_7_0_7_0_7_{src}', pyramid_0_7_0_7_0_7)

        pyramid_0_7_0_7_1_5 = [img, layer_1_0_7, layer_2_0_7_0_7, layer_3_0_7_0_7_1_5]
        save_pyramids(path, f'0_7_0_7_1_5_{src}', pyramid_0_7_0_7_1_5)

        pyramid_0_7_1_5_0_7 = [img, layer_1_0_7, layer_2_0_7_1_5, layer_3_0_7_1_5_0_7]
        save_pyramids(path, f'0_7_1_5_0_7_{src}', pyramid_0_7_1_5_0_7)

        pyramid_0_7_1_5_1_5 = [img, layer_1_0_7, layer_2_0_7_1_5, layer_3_0_7_1_5_1_5]
        save_pyramids(path, f'0_7_1_5_1_5_{src}', pyramid_0_7_1_5_1_5)

        pyramid_1_5_0_7_0_7 = [img, layer_1_1_5, layer_2_1_5_0_7, layer_3_1_5_0_7_0_7]
        save_pyramids(path, f'1_5_0_7_0_7_{src}', pyramid_1_5_0_7_0_7)

        pyramid_1_5_0_7_1_5 = [img, layer_1_1_5, layer_2_1_5_0_7, layer_3_1_5_0_7_1_5]
        save_pyramids(path, f'1_5_0_7_1_5_{src}', pyramid_1_5_0_7_1_5)

        pyramid_1_5_1_5_0_7 = [img, layer_1_1_5, layer_2_1_5_1_5, layer_3_1_5_1_5_0_7]
        save_pyramids(path, f'1_5_1_5_0_7_{src}', pyramid_1_5_1_5_0_7)

        pyramid_1_5_1_5_1_5 = [img, layer_1_1_5, layer_2_1_5_1_5, layer_3_1_5_1_5_1_5]
        save_pyramids(path, f'1_5_1_5_1_5_{src}', pyramid_1_5_1_5_1_5)

        print('> pyramids saved')

    except Exception as ex:
        print(f'> Test corrupted by exception: {ex}.')


def restore_sobel():
    for src in [
        '_0_7_0_7_0_7_bears',
        '_0_7_0_7_1_5_bears',
        '_0_7_1_5_0_7_bears',
        '_0_7_1_5_1_5_bears',
        '_1_5_0_7_0_7_bears',
        '_1_5_0_7_1_5_bears',
        '_1_5_1_5_0_7_bears',
        '_1_5_1_5_1_5_bears',
        '_clear_bears'
    ]:
        gray = cv.imread(f'./data/lr3/test/gray{src}.png', 0)
        sobel = Sobel.make_sobel_convolution(gray)
        cv.imwrite(f'./data/lr3/test/sobel{src}.png', sobel)


if __name__ == '__main__':
    # test_lr_3('./data/lr3/test/', 'bears.png')
    restore_sobel()

    """
        # lr1.run()
    
        # 1. test params
        path = './data/lr2/'
        noise_k = 0.05
        box_blur_kernel = blur.Kernels.MIDDLE_KERNEL
        sigma = 1.4
        radius = 4
    
        # 2. read image
        # img = cv.imread(path + src, cv.IMREAD_COLOR)
        # print('> image read')
        # # plot.show_plot_image(img)
        # kernel = list(map(lambda row: [item / 26 for item in row], kernel))
        img = cv.imread(path + src, cv.IMREAD_COLOR)
        img_n = noise.sp_noise(img, 0.05)
        cv.imwrite(f'{path}_noise_{src}', img_n)
        print('noise done')
        img_b = BlurProcessor.make_convolution(img_n, Kernel.make_kernel_normalization(Kernel.BIG_KERNEL))
        cv.imwrite(f'{path}_blur_{src}', img_b)
        print('blur done')
        img_log = BlurProcessor.make_convolution(img_b, Kernel.create_LoG_kernel(2, 7))
        cv.imwrite(f'{path}_LoG_{src}', img_log)
        print('LoG done')
    
        # 3. add noise
        # img_n = noise.sp_noise(img, noise_k)
        # cv.imwrite(f'{path}noise_{noise_k}_{src}', img_n)
        # print('> image with noise wrote')
        # plot.show_plot_image(img_n)
    
        # 4. LoG
        # img_f = blur.LoG(img, sigma, radius)
        # cv.imwrite(f'{path}LoG_blur_sigma{sigma}_r{radius}_{src}', img_f)
        # print('> LoG blur image wrote')
    
        # 5. box blur and sharpening
        # img_f = blur.box_blur(img_n, box_blur_kernel)
        # cv.imwrite(f'{path}box_blur_x{len(blur.Kernels.MIDDLE_KERNEL)}_{src}', img_f)
        # print('> box blur image wrote')
    
        # img_s = blur.laplacian_sharpening(img_f)
        # cv.imwrite(f'{path}sharpening_box_{src}', img_s)
        # print('> sharpening image wrote')
    
        # 6. gaussian blur and sharpening
        # img_f = blur.gaussian_blur(img_n, sigma, radius)
        # cv.imwrite(f'{path}gaussian_blur_sigma{sigma}_r{radius}_{src}', img_f)
        # print('> gaussian blur image wrote')
    
        # img_s = blur.laplacian_sharpening(img_f)
        # cv.imwrite(f'{path}sharpening_gaussian_{src}', img_s)
        # print('> sharpening image wrote')
    
        # 7. median blur and sharpening
        # img_f = blur.median_blur(img_n, radius)
        # cv.imwrite(f'{path}median_blur_r{radius}_{src}', img_f)
        # print('> median blur image wrote')
    
        # img_s = blur.laplacian_sharpening(img_f)
        # cv.imwrite(f'{path}sharpening_median_{src}', img_s)
        # print('> sharpening image wrote')
    
    
    
        # # img_f = blur.gaussian_blur(img_n, 1.5, 3)
        # img_f = blur.median_blur(img_n, 5)
        # plot.show_plot_image(img_f)
        # img_s = blur.laplacian_sharpening(img_f)
        # plot.show_plot_image(img_s)
    """