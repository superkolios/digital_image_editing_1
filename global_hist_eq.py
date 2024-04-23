import numpy as np


def get_equalization_transform_of_img( img_array: np.ndarray,) -> np.ndarray:
    histogram = np.zeros(256)
    for i in img_array:
        for j in i:
            histogram[j] += 1
    num_of_pixeles = img_array.size
    histogram = histogram / num_of_pixeles
    yk = np.cumsum(histogram)
    y0 = yk[0]
    if (y0 == 1):
        # todo: return a defult transform so program continues execution
        print('region has only zeros')
        exit()
    yk = np.round((yk - y0)/(1 - y0) * 255)
    return yk.astype(np.uint8)


def perform_global_hist_equalization(img_array: np.ndarray,) -> np.ndarray:
    yk = get_equalization_transform_of_img(img_array)
    return yk[img_array]
