import numpy as np


def get_equalization_transform_of_img( img_array: np.ndarray,) -> np.ndarray:
    # calculate histogram
    histogram = np.zeros(256)
    for row in img_array:
        for pixel_value in row:
            histogram[pixel_value] += 1

    # divide each ellemnt with the number of pixels to get the probability
    num_of_pixeles = img_array.size
    histogram = histogram / num_of_pixeles

    # calculate cumulative distribution function
    y = np.cumsum(histogram)
    y0 = y[0]

    # check if a region pixel values has only zeros
    if (y0 == 1):
        # Generate a default transformation that doesn't alter pixel values.
        print('region has only zeros')
        equalization_transform = np.arange(256)
        return equalization_transform
    
    equalization_transform = np.round((y - y0)/(1 - y0) * 255)
    equalization_transform = equalization_transform.astype(np.uint8)
    return equalization_transform


def perform_global_hist_equalization(img_array: np.ndarray,) -> np.ndarray:
    yk = get_equalization_transform_of_img(img_array)
    equalized_img = yk[img_array]
    return equalized_img
