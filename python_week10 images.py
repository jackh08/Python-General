#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Exercise – Week X
Data Programming With Python – Fall / 2017
Image Processing
"""

from skimage.filters import threshold_otsu, threshold_mean, threshold_li
from skimage.morphology import reconstruction
from skimage.segmentation import clear_border
from skimage.segmentation import mark_boundaries
from skimage import data, io, filters
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import roberts, sobel, scharr, prewitt
#============================= Load the "cell.jpg" image =======================#
original_image = io.imread('C:\Users\jackh\OneDrive\Documents\College\Python\Images\cell.jpg')
#================================= Processing ==================================#
 #1--------- Edge detection --------#
edge_sobel = sobel(original_image)

 #2--------- Thresholding ----------#
thresh = threshold_li(edge_sobel)

 #3--------- Binarization ----------#
binary_image = edge_sobel > thresh

plt.imshow(filled_image)

### reconstruction
 #4-------------- Filling in the image with white colour ------------------#
binary_image_copy = np.copy(binary_image)
binary_image_copy[1:-1, 1:-1] = 1
filled_image = reconstruction(binary_image_copy,binary_image, method='erosion')

plt.imshow(binary_image)
plt.imshow(filled_image)
# Reconstruction by erosion reconstructs dark regions in a grayscale image and
# holes in a binary image. Neighboring pixels are reconstructed by spreading the
# darkness value. Reconstruction by erosion starts with the minimal valued pixels
# of the marker and reconstructs the neighboring pixels ranging from the minimal
# valued pixel to the image maximum value.

### eliminate redundant part
 #5----------- eliminate the redundant part of image ----------------------#
cleared_border_image = clear_border(filled_image, buffer_size=16)

plt.imshow(cleared_border_image)

### clearing up
 #6-- Heuristically clearing up (as much as possible) the white dots ------#
cleaned_of_white_dots_image = np.copy(cleared_border_image)
cleaned_of_white_dots_image[200:300,0:300] = 0
cleaned_of_white_dots_image[:,0:100] = 0
cleaned_of_white_dots_image[:,300::] = 0

plt.imshow(cleared_border_image)
plt.imshow(cleaned_of_white_dots_image)

### Plot all sorts!
 #7------------- Highlight the cell borders in the main image -------------#
label_img = cleaned_of_white_dots_image > 0
highlighted_border_image = mark_boundaries(original_image,label_img,color=(0.5, 1, 0.5))
#=================================== Plotting ==================================#
plt.subplot(1,7,1)
plt.imshow(original_image, cmap=plt.cm.gray)
plt.title('Original Image')
plt.subplot(1,7,2)
plt.imshow(edge_sobel, cmap=plt.cm.gray)
plt.title('Edge Detected Image')
plt.subplot(1,7,3)
plt.imshow(binary_image, cmap=plt.cm.gray)
plt.title('Binarized Image')
plt.subplot(1,7,4)
plt.imshow(filled_image, cmap=plt.cm.gray)
plt.title('Filled Image')
plt.subplot(1,7,5)
plt.imshow(cleared_border_image, cmap=plt.cm.gray)
plt.title('Cleared border Image')
plt.subplot(1,7,6)
plt.imshow(cleaned_of_white_dots_image, cmap=plt.cm.gray)
plt.title('Eliminated dots Image')
plt.subplot(1,7,7)
plt.imshow(highlighted_border_image , cmap=plt.cm.gray)
plt.title('Border-Highlighted Image')
plt.show()

# Image Processing
### Dice example
# From the data folder import the image called dice.jpg, and do the followings :
# 1. Read the input image, convert it to grayscale, and blur it slightly.
# 2. Use simple fixed-level thresholding to convert the grayscale image to a binary image.
# 3. Find contours corresponding to the outlines of each dice.
# 4. Print information on how many dices can be found in the image.
# 5. For illustrative purposes, mark the centre position of each dice in the image grid so we
# can visualize the results.

import matplotlib.pyplot as plt
import numpy as np
from skimage import data, io, filters
from skimage.color import rgb2grey
from skimage.filters import threshold_otsu, threshold_mean,threshold_li,threshold_yen,threshold_minimum
from skimage.morphology import reconstruction
from skimage.measure import find_contours
from skimage.feature import blob_dog, blob_log, blob_doh
#================= Importing the ‘Dice.jpg’ image ==============#
original_image = io.imread('C:\Users\jackh\OneDrive\Documents\College\Python\Images\dice.jpg')
#======================= Processing ============================# takes 1 minute
Black_White_image = rgb2grey(original_image)
thresh = threshold_minimum(Black_White_image)
binary_image = Black_White_image > thresh
contours1 = find_contours(binary_image ,0.1)
binary_image_copy = np.copy(binary_image)
binary_image_copy[1:-1, 1:-1] = 1
filled_image = reconstruction(binary_image_copy,binary_image,method='erosion')
filled_image [0:100,:] = 0
filled_image [:,0:200] = 0
contours2 = find_contours(filled_image ,0.1)
Blob_centres = blob_log(filled_image, min_sigma=100)


### count dice
#===================== Plotting & Printing =======================#
print('The Total Number of Dices in Image = ', len(Blob_centres))
plt.subplot(2,4,1)
plt.imshow(original_image, cmap=plt.cm.gray)
plt.title('Original Image')
plt.subplot(2,4,2)
plt.imshow(Black_White_image, cmap=plt.cm.gray)
plt.title('Black & White Image')

plt.subplot(2,4,3)
plt.imshow(binary_image, cmap=plt.cm.gray)
plt.title('Binarized Image')
plt.subplot(2,4,4)
plt.imshow(binary_image, cmap=plt.cm.gray)


for n, contour in enumerate(contours1):
    plt.plot(contour[:, 1], contour[:, 0], 'r-', linewidth=2.5)
plt.title('Contoured Image')
plt.subplot(2,4,5)
plt.imshow(filled_image, cmap=plt.cm.gray)
plt.title('Filled Image')
plt.subplot(2,4,6)
plt.imshow(filled_image , cmap=plt.cm.gray)
for n, contour in enumerate(contours2):
    plt.plot(contour[:, 1], contour[:, 0], 'y-', linewidth=5.5)
plt.title('Outer Contoured Image')
plt.subplot(2,4,7)
plt.imshow(original_image , cmap=plt.cm.gray)
for n, contour in enumerate(contours2):
    plt.plot(contour[:, 1], contour[:, 0], 'b-', linewidth=5.5)
plt.title('Outer Contour on Original Image')
plt.subplot(2,4,8)
plt.imshow(Blob_centres, cmap=plt.cm.gray)
for i, Blob in enumerate(Blob_centres):
    plt.plot(Blob_centres[:, 1], Blob_centres[:, 0], 'go', markersize=15)
plt.grid()
plt.title('Binary Large Objects (BLOB)')
plt.show()

### Lecture

import numpy as np
import matplotlib.pyplot as plt
Chess = np.zeros((7,7))
Chess[ ::2,1::2] = 1
Chess[1::2, ::2] = 1
plt.imshow(Chess)
plt.show()

from skimage import io
import matplotlib.pyplot as plt
my_image = io.imread('C:\Users\jackh\OneDrive\Documents\College\Python\Images\pose.jpg')
plt.imshow(my_image)
plt.show()

### rotate an image

from skimage import data
from skimage.transform import rotate
image = data.horse()
plt.subplot(2,1,1)
plt.imshow(image)
rotated_image = rotate(image, 30)
plt.subplot(2,1,2)
plt.imshow(rotated_image)
plt.show()

### resize

from skimage import data
from skimage.transform import resize
image = data.horse()
plt.subplot(2,1,1)
plt.imshow(image)
resized_image = resize(image, (150,150)) #you loose the resolution, as the original is 400x400
plt.subplot(2,1,2)
plt.imshow(resized_image)
plt.show()

### rescale

from skimage import data
from skimage.transform import rescale
image = data.horse()
plt.subplot(2,1,1)
plt.imshow(image)
rescaled_image = rescale(image, 0.5)
plt.subplot(2,1,2)
plt.imshow(rescaled_image)
plt.show()

### change colour

import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2hsv
red_pixel_rgb = np.array([[[255, 0, 0]]], dtype=np.uint8)
rgb2hsv(red_pixel_rgb)
print(rgb2hsv(red_pixel_rgb))
plt.subplot(2,1,1)
plt.imshow(red_pixel_rgb)
plt.subplot(2,1,2)
plt.imshow(rgb2hsv(red_pixel_rgb))
plt.show()

### change colour to grey scale

import matplotlib.pyplot as plt
from skimage.color import rgb2grey
from skimage import data
my_image = data.astronaut()
plt.subplot(2,1,1)
plt.imshow(my_image)
plt.subplot(2,1,2)
plt.imshow(rgb2grey(my_image))
plt.show()

### drawing shapes

from skimage.draw import polygon, ellipse
import numpy as np
import matplotlib.pyplot as plt
#=============== Create Shapes ============#
my_image = np.zeros((600, 600, 3), dtype=np.double)
#---- Polygon ---- #
Poly = np.array([[300, 300], [480, 320], [380, 430], [220, 590], [300, 300]])
row_poly = Poly[:, 0]
column_poly = Poly[:, 1]
r_p, c_c = polygon(row_poly, column_poly)
my_image[r_p, c_c, 1] = 1
#---- Ellipse ---- #
r_e, c_e = ellipse(300, 300, 100, 220, rotation = 60)
my_image[r_e, c_e, 2] = 1
#================== Plotting =================#
plt.imshow(my_image)
plt.show()

### edge detection
import matplotlib.pyplot as plt
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage import data
#============================#
image = data.coins()
edge_sobel = sobel(image)
edge_roberts = roberts(image)
edge_scharr = scharr(image)
edge_prewitt = prewitt(image)
#============================#
plt.subplot(1,5,1)
plt.imshow(image) #original image
plt.subplot(1,5,2)
plt.imshow(edge_sobel ,cmap=plt.cm.gray)
plt.subplot(1,5,3)
plt.imshow(edge_roberts,cmap=plt.cm.gray)
plt.subplot(1,5,4)
plt.imshow(edge_scharr ,cmap=plt.cm.gray)
plt.subplot(1,5,5)
plt.imshow(edge_prewitt,cmap=plt.cm.gray)

### edge and contour detection

from skimage import data
from skimage.measure import find_contours
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
#============= Construct some test data ==============#
x, y = np.ogrid[-np.pi:np.pi:100j, -np.pi:np.pi:100j]
r = np.sin(np.exp((np.sin(x)**3 + np.cos(y)**2)))
#====== Find contours at a constant value of 0.8 =====#
contours = find_contours(r, 0.8)
#=== Display the image and plot all contours found ===#
plt.imshow(r, interpolation='nearest', cmap=plt.cm.gray)
for n, contour in enumerate(contours):
    plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
plt.show()

### snake 

from skimage import data
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from skimage.segmentation import active_contour
#================================================#
image = data.astronaut()
image = rgb2grey(image)
#===== Defining initial Snake as a circle =======#
s = np.linspace(0, 2*np.pi, 400)
x = 220 + 100*np.cos(s)
y = 100 + 100*np.sin(s)
initial_snake = np.array([x, y]).T
#====== Applying forces for Snake to grasp ======#
snake = active_contour(gaussian(image, 3),initial_snake,alpha=0.015,beta=10,gamma=0.001)
#================== Plotting ====================#
plt.imshow(image, cmap=plt.cm.gray)
plt.plot(initial_snake[:, 0], initial_snake[:, 1], '--r', lw=3)
plt.plot(snake[:, 0], snake[:, 1], '-c', lw=5)
plt.show()

### template matching

import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import match_template
from skimage import data
#=================== Import Dataset & Template ====================#
image = data.coins()
coin = image[170:220, 75:130] # Template
hight_coin, width_coin = coin.shape # Get the size of template
#======================= Template Matching =======================#
result = match_template(image, coin)
pixel = np.unravel_index(np.argmax(result), result.shape)
x, y = pixel[::-1]
#=========================== Plotting =============================#
ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)
ax1.imshow(coin, cmap=plt.cm.gray)
ax2.imshow(image, cmap=plt.cm.gray)
rect = plt.Rectangle((x, y), width_coin , hight_coin, edgecolor='r', facecolor='none')
ax2.add_patch(rect) # add a patch on that part of image which contains template
plt.show()

### denoising

from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,denoise_wavelet)
from skimage import data, io
from skimage.util import random_noise
#================== Import Image =====================#
original_image = io.imread('C:\Users\jackh\OneDrive\Documents\College\Python\Images\pose.jpg')
# The image is provided for you in this week’s Learning Material folder
#================ Add Random Noise ===================#
noisy_image = random_noise(original_image, var=0.22**2)
#================== Denoising ========================#
denoised_image_tv = denoise_tv_chambolle(noisy_image, weight=0.1, multichannel=True)
denoised_image_wv = denoise_wavelet(noisy_image, multichannel=True)
#denoised_image_bi = denoise_bilateral(noisy_image, sigma_color=0.15, sigma_spatial=35, multichannel=True) #!!!Bilateral filter is very slow to run !!!
#=================== Plotting ========================#
plt.subplot(1,5,1)
plt.imshow(original_image)
plt.title('Original')
plt.subplot(1,5,2)
plt.imshow(noisy_image)
plt.title('Noisy Image')
plt.subplot(1,5,3)
plt.imshow(denoised_image_tv)
plt.title('TV Chambolle denoised image')
plt.subplot(1,5,4)
plt.imshow(denoised_image_wv)
plt.title('Wavelet denoised image')
plt.show()

### Thresholding

from skimage.filters import threshold_otsu
from skimage.filters import threshold_mean
from skimage.filters import threshold_li
from skimage.filters import threshold_yen
from skimage.filters import threshold_minimum
from skimage.filters import threshold_isodata

GreyScale_image = rgb2grey(my_image)

Thresh = threshold_otsu(GreyScale_image)
Thresh = threshold_mean(GreyScale_image)
Thresh = threshold_li(GreyScale_image)
Thresh = threshold_yen(GreyScale_image)
Thresh = threshold_minimum(GreyScale_image)
Thresh = threshold_isodata(GreyScale_image)
Binary_Image = GreyScale_image > Thresh

plt.show(Binary_Image)
