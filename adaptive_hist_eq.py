import numpy as np
from global_hist_eq import *

def calculate_eq_transformations_of_regions(img_array: np.ndarray, region_len_h: int,
                                            region_len_w: int) -> dict[tuple, np.ndarray]:
    # get image dimantions
    img_height, img_width = img_array.shape
    # check if image dimansions are divisible by region lengths
    assert img_width % region_len_w == 0, "image width is not divisible by region_len_w"
    assert img_height % region_len_h == 0, "image height is not divisible by region_len_h"
    
    # calculate the equalization transformation for each region in the image
    region_to_eq_transform = dict()
    for y in range(0, img_height, region_len_h):
        for x in range(0, img_width, region_len_w):
            region_to_eq_transform[(x,y)] = get_equalization_transform_of_img(img_array[y:y+region_len_h, x:x+region_len_w])
    return region_to_eq_transform


def perform_adaptive_hist_equalization(img_array: np.ndarray, region_len_h: int,
                                       region_len_w: int) -> np.ndarray:
    
    region_to_eq_transform = calculate_eq_transformations_of_regions(img_array, region_len_h, region_len_w)
    # get image dimantions
    img_height, img_width = img_array.shape
    # initialize equalized_img
    equalized_img = np.zeros((img_height, img_width), dtype=np.uint8)

    center_x = region_len_w/2
    center_y = region_len_h/2
    
    # iterate for each pixel
    for y in range(img_height):
        for x in range(img_width):
            # find the region the pixel belongs
            region_x = x - (x % region_len_w)
            region_y = y - (y % region_len_h)
            
            pixel_value = img_array[y,x]

            # check if the pixel is at the outer area
            if (y < region_len_h/2 or y >= img_height - (region_len_h)/2 or x < region_len_w/2 or x >= img_width - (region_len_w)/2):
                # apply transformation without interpolation
                equalized_img[y, x] = region_to_eq_transform[(region_x, region_y)][pixel_value]
            else:
                # apply transformation with interpolation

                # find the relative position starting from top left corner 
                relative_X = x % region_len_w
                relative_y = y % region_len_h
                
                # find neigtboring regions based on the position to the center of the region
                # and calculate weights
                # up: y position of regions above the pixel
                # down: y position of regions below the pixel
                # left: x position of regions left of the pixel
                # right: x position of regions right of the pixel
                if (relative_X < center_x):
                    left = region_x - region_len_w
                    right = region_x
                    a = (relative_X + region_len_w/2)/region_len_w
                else:
                    left = region_x
                    right = region_x + region_len_w
                    a = (relative_X - region_len_w/2)/region_len_w

                if (relative_y < center_y):
                    up = region_y - region_len_h
                    down = region_y
                    b = (relative_y + region_len_h/2)/region_len_h
                else:
                    up = region_y
                    down = region_y + region_len_h
                    b = (relative_y - region_len_h/2)/region_len_h
                
                # Retrieve the equalization transformations for the four neighboring regions.
                # ul: up left
                # ur: up right
                # dl: down left
                # dr: down right
                ul = region_to_eq_transform[(left, up)][pixel_value]
                ur = region_to_eq_transform[(right, up)][pixel_value]
                dl = region_to_eq_transform[(left, down)][pixel_value]
                dr = region_to_eq_transform[(right, down)][pixel_value]
                # apply interpolation
                equalized_img[y,x] = (1 - a)*(1 - b)*ul + (1 - a)*b*dl + a*(1 - b)*ur + a*b*dr
    return equalized_img

# adaptive hist equalization without interpolation
def perform_adaptive_hist_equalization_no_interpolation(img_array: np.ndarray, region_len_h: int,
                                              region_len_w: int) -> np.ndarray:
    
    region_to_eq_transform = calculate_eq_transformations_of_regions(img_array, region_len_h, region_len_w)
    # get image dimantions
    img_height, img_width = img_array.shape
    # initialize equalized_img
    equalized_img = np.zeros((img_height, img_width), dtype=np.uint8)
    
    # iterate for each pixel
    for y in range(img_height):
        for x in range(img_width):
            # find the region the pixel belongs
            region_x = x - (x % region_len_w)
            region_y = y - (y % region_len_h)
            pixel_value = img_array[y,x]
            
            # apply transformation without interpolation
            equalized_img[y, x] = region_to_eq_transform[(region_x, region_y)][pixel_value]
    return equalized_img
