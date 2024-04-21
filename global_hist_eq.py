from PIL import Image
import numpy as np
from matplotlib import pyplot as plt



def get_equalization_transform_of_img( img_array: np.ndarray,) -> np.ndarray:
    histogram = np.zeros(256)
    for i in img_array:
        for j in i:
            histogram[j] += 1
    num_of_pixeles = img_array.shape[0] * img_array.shape[1]
    uk = histogram/ num_of_pixeles
    return uk
    


def perform_global_hist_equalization(img_array: np.ndarray,) -> np.ndarray:
    uk = get_equalization_transform_of_img(img_array)

    
    



# set the filepath to the image file
filename = "input_img.png"
# read the image into a PIL entity
img = Image.open(fp=filename)
# keep only the Luminance component of the image
bw_img = img.convert("L")
# obtain the underlying np array
img_array = np.array(bw_img)

uk = get_equalization_transform_of_img(img_array)
print(uk.shape)
