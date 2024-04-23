from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

from global_hist_eq import *
from adaptive_hist_eq import *

# set the filepath to the image file
filename = "input_img.png"
# read the image into a PIL entity
img = Image.open(fp=filename)
# keep only the Luminance component of the image
bw_img = img.convert("L")
# obtain the underlying np array
img_array = np.array(bw_img)

# question 1
equalization_transform = get_equalization_transform_of_img(img_array)
fig_1 = plt.figure()
fig_1.canvas.manager.set_window_title('Equalization transform')
plt.title('Equalization transform of image')
plt.xlabel('input value')
plt.ylabel('output value')
plt.stairs(equalization_transform, baseline=None)


# question 2
fig, axs = plt.subplots(2, 2)
fig.subplots_adjust(wspace=0.5)
fig.canvas.manager.set_window_title('Global histogram equalization')

axs[0,0].imshow(img_array, cmap='gray')
axs[0,0].set_title('Original image')
axs[0,0].axis('off')
axs[1,0].hist(img_array.ravel(), bins=256, range=(0, 256))
axs[1,0].set_title('Histogram')
axs[1,0].set_ylabel('count')
axs[1,0].set_xlabel('brightness')

global_eq_img = perform_global_hist_equalization(img_array)
axs[0,1].imshow(global_eq_img, cmap='gray')
axs[0,1].set_title('Global equalised image')
axs[0,1].axis('off')
axs[1,1].hist(global_eq_img.ravel(), bins=256, range=(0, 256))
axs[1,1].set_title('Histogram')
axs[1,1].set_ylabel('count')
axs[1,1].set_xlabel('brightness')


# question 3
fig, axs = plt.subplots(2, 3)
fig.set_figwidth(10)
fig.subplots_adjust(wspace=0.5)
fig.canvas.manager.set_window_title('Adaptive histogram equalization')

axs[0,0].imshow(img_array, cmap='gray')
axs[0,0].set_title('Original image')
axs[0,0].axis('off')
axs[1,0].hist(img_array.ravel(), bins=256, range=(0, 256))
axs[1,0].set_title('Histogram')
axs[1,0].set_ylabel('count')
axs[1,0].set_xlabel('brightness')

adaptive_eq_img = perform_adaptive_hist_equalization(img_array, region_len_h=36, region_len_w=48)
axs[0,1].imshow(adaptive_eq_img, cmap='gray')
axs[0,1].set_title('Adaptive histogram \n equalised image')
axs[0,1].axis('off')
axs[1,1].hist(adaptive_eq_img.ravel(), bins=256, range=(0, 256))
axs[1,1].set_title('Histogram')
axs[1,1].set_ylabel('count')
axs[1,1].set_xlabel('brightness')

adaptive_eq_img_no_interpolation = perform_adaptive_hist_equalization_simple(img_array, region_len_h=36, region_len_w=48)
axs[0,2].imshow(adaptive_eq_img_no_interpolation, cmap='gray')
axs[0,2].set_title('Adaptive histogram equalised \n image without interpolation')
axs[0,2].axis('off')
axs[1,2].hist(adaptive_eq_img_no_interpolation.ravel(), bins=256, range=(0, 256))
axs[1,2].set_title('Histogram')
axs[1,2].set_ylabel('count')
axs[1,2].set_xlabel('brightness')

plt.show()
