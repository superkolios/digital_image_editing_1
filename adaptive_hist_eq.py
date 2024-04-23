import numpy as np
from global_hist_eq import *

def calculate_eq_transformations_of_regions(img_array: np.ndarray, region_len_h: int,
                                            region_len_w: int) -> dict[tuple, np.ndarray]:

    img_height, img_width = img_array.shape
    # check if image dimansions are divisible by region lengths
    assert img_width % region_len_w == 0, "image width is not divisible by region_len_w"
    assert img_height % region_len_h == 0, "image height is not divisible by region_len_h"
    
    region_to_eq_transform = dict()
    for y in range(0, img_height, region_len_h):
        for x in range(0, img_width, region_len_w):
            region_to_eq_transform[(x,y)] = get_equalization_transform_of_img(img_array[y:y+region_len_h, x:x+region_len_w])
    return region_to_eq_transform


def perform_adaptive_hist_equalization(img_array: np.ndarray, region_len_h: int,
                                       region_len_w: int) -> np.ndarray:
    
    region_to_eq_transform = calculate_eq_transformations_of_regions(img_array, region_len_h, region_len_w)
    img_height, img_width = img_array.shape
    equalized_img = np.zeros((img_height, img_width), dtype=np.uint8)

    center_x = region_len_w/2
    center_y = region_len_h/2
    for y in range(img_height):
        for x in range(img_width):
            # find region
            region_x = x - (x % region_len_w)
            region_y = y - (y % region_len_h)
            pixel_value = img_array[y,x]

            # check if pixel is an outer point
            if (y < region_len_h/2 or y >= img_height - (region_len_h)/2 or x < region_len_w/2 or x >= img_width - (region_len_w)/2):
                equalized_img[y, x] = region_to_eq_transform[(region_x, region_y)][pixel_value]
            else:
                relative_X = x % region_len_w
                relative_y = y % region_len_h
                
                # check position relative to the center of the region
                # ul: up left
                # ur: up right
                # dl: down left
                # dr: down right
                if (relative_X < center_x and relative_y < center_y):
                    ul = region_to_eq_transform[(region_x - region_len_w, region_y - region_len_h)][pixel_value]
                    ur = region_to_eq_transform[(region_x, region_y - region_len_h)][pixel_value]
                    dl = region_to_eq_transform[(region_x - region_len_w, region_y)][pixel_value]
                    dr = region_to_eq_transform[(region_x, region_y)][pixel_value]
                    a = (relative_X + region_len_w/2)/region_len_w
                    b = (relative_y + region_len_h/2)/region_len_h
                elif (relative_X < center_x and relative_y >= center_y):
                    ul = region_to_eq_transform[(region_x - region_len_w, region_y)][pixel_value]
                    ur = region_to_eq_transform[(region_x, region_y)][pixel_value]
                    dl = region_to_eq_transform[(region_x - region_len_w, region_y + region_len_h)][pixel_value]
                    dr = region_to_eq_transform[(region_x, region_y + region_len_h)][pixel_value]
                    a = (relative_X + region_len_w/2)/region_len_w
                    b = (relative_y - region_len_h/2)/region_len_h
                elif (relative_X >= center_x and relative_y < center_y):
                    ul = region_to_eq_transform[(region_x, region_y - region_len_h)][pixel_value]
                    ur = region_to_eq_transform[(region_x + region_len_w, region_y - region_len_h)][pixel_value]
                    dl = region_to_eq_transform[(region_x, region_y)][pixel_value]
                    dr = region_to_eq_transform[(region_x + region_len_w, region_y)][pixel_value]
                    a = (relative_X - region_len_w/2)/region_len_w
                    b = (relative_y + region_len_h/2)/region_len_h
                else:
                    ul = region_to_eq_transform[(region_x, region_y)][pixel_value]
                    ur = region_to_eq_transform[(region_x + region_len_w, region_y)][pixel_value]
                    dl = region_to_eq_transform[(region_x, region_y + region_len_h)][pixel_value]
                    dr = region_to_eq_transform[(region_x + region_len_w, region_y + region_len_h)][pixel_value]
                    a = (relative_X - region_len_w/2)/region_len_w
                    b = (relative_y - region_len_h/2)/region_len_h

                equalized_img[y,x] = (1 - a)*(1 - b)*ul + (1 - a)*b*dl + a*(1 - b)*ur + a*b*dr
    return equalized_img

# adaptive hist equalization without interpolation
def perform_adaptive_hist_equalization_simple(img_array: np.ndarray, region_len_h: int,
                                              region_len_w: int) -> np.ndarray:
    
    region_to_eq_transform = calculate_eq_transformations_of_regions(img_array, region_len_h, region_len_w)
    img_height, img_width = img_array.shape
    equalized_img = np.zeros((img_height, img_width), dtype=np.uint8)

    for y in range(img_height):
        for x in range(img_width):
            # find region
            region_x = x - (x % region_len_w)
            region_y = y - (y % region_len_h)
            pixel_value = img_array[y,x]
            equalized_img[y, x] = region_to_eq_transform[(region_x, region_y)][pixel_value]
    return equalized_img
