# VIP :: Assignment #2
### Olga Iarygina, hwk263

### importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread # for image importing
import cv2

from scipy.ndimage import gaussian_filter # for gaussian filtering
from scipy.ndimage import gaussian_laplace # for laplacian-gaussian

plt.rcParams['figure.figsize'] = [12, 12]

# 0. Importing the image

img = imread('lenna.jpg')

# checking that everything is imported correctly
plt.imshow(img, cmap='Greys_r')
plt.gcf()
plt.axis('off')
plt.show()

# 1. Gaussian filtering

# For each of the sigmas I implement the gaussian_filter function
# And then plot the result to see the differences in blurring

### 1.1 Sigma = 1
gaus_sigma_1 = gaussian_filter(img, sigma = 1)

fig = plt.figure()
plt.gray()  
ax1 = fig.add_subplot(121)  
ax2 = fig.add_subplot(122)  
ax1.imshow(img)
ax2.imshow(gaus_sigma_1)
plt.show()

### 1.2 Sigma = 2
gaus_sigma_2 = gaussian_filter(img, sigma = 2)

fig = plt.figure()
plt.gray()  
ax1 = fig.add_subplot(121)  
ax2 = fig.add_subplot(122)  
ax1.imshow(img)
ax2.imshow(gaus_sigma_2)
plt.show()

### 1.3 Sigma = 4
gaus_sigma_4 = gaussian_filter(img, sigma = 4)

fig = plt.figure()
plt.gray()  
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122) 
ax1.imshow(img)
ax2.imshow(gaus_sigma_4)
plt.show()

### 1.4 Sigma = 8
gaus_sigma_8 = gaussian_filter(img, sigma = 8)

fig = plt.figure()
plt.gray() 
ax1 = fig.add_subplot(121)  
ax2 = fig.add_subplot(122)  
ax1.imshow(img)
ax2.imshow(gaus_sigma_8)
plt.show()

# 2. Gradient magnitude with Gaussian derivatives

# For each of the sigmas I compute x-derivative and y-derivative
# Using the same gaussian_filter function as before, but specifying the order of axes
# Then I plot the resulting images

### 2.1 Sigma = 1
x_resp_1 = gaussian_filter(img, sigma = 1, order = [1, 0], output = np.float64)
y_resp_1 = gaussian_filter(img, sigma = 1, order = [0, 1], output = np.float64)

fig = plt.figure()
plt.gray()  
ax1 = fig.add_subplot(121)  
ax2 = fig.add_subplot(122)  
ax1.imshow(x_resp_1)
ax2.imshow(y_resp_1)
ax1.set_title('Response on Gaussian x-derivative filter, sigma = 1')
ax2.set_title('Response on Gaussian y-derivative filter, sigma = 1')
plt.show()

### 2.2 Sigma = 2
x_resp_2 = gaussian_filter(img, sigma = 2, order = [1, 0], output = np.float64)
y_resp_2 = gaussian_filter(img, sigma = 2, order = [0, 1], output = np.float64)

fig = plt.figure()
plt.gray()  
ax1 = fig.add_subplot(121)  
ax2 = fig.add_subplot(122) 
ax1.imshow(x_resp_2)
ax2.imshow(y_resp_2)
ax1.set_title('Response on Gaussian x-derivative filter, sigma = 2')
ax2.set_title('Response on Gaussian y-derivative filter, sigma = 2')
plt.show()

### 2.3 Sigma = 3
x_resp_4 = gaussian_filter(img, sigma = 4, order = [1, 0], output = np.float64)
y_resp_4 = gaussian_filter(img, sigma = 4, order = [0, 1], output = np.float64)

fig = plt.figure()
plt.gray()  
ax1 = fig.add_subplot(121)  
ax2 = fig.add_subplot(122)  
ax1.imshow(x_resp_4)
ax2.imshow(y_resp_4)
ax1.set_title('Response on Gaussian x-derivative filter, sigma = 4')
ax2.set_title('Response on Gaussian y-derivative filter, sigma = 4')
plt.show()

### 2.4 Sigma = 4
x_resp_8 = gaussian_filter(img, sigma = 8, order = [1, 0], output = np.float64)
y_resp_8 = gaussian_filter(img, sigma = 8, order = [0, 1], output = np.float64)

fig = plt.figure()
plt.gray()  
ax1 = fig.add_subplot(121) 
ax2 = fig.add_subplot(122)  
ax1.imshow(x_resp_8)
ax2.imshow(y_resp_8)
ax1.set_title('Response on Gaussian x-derivative filter, sigma = 8')
ax2.set_title('Response on Gaussian y-derivative filter, sigma = 8')
plt.show()

# 3. Laplacian-Gaussian filtering

# For each of the sigmas I apply Laplacian of Gaussian
# And plot the results altogether

fig = plt.figure()
plt.gray()  

ax1 = fig.add_subplot(221)  
ax2 = fig.add_subplot(222)  
ax3 = fig.add_subplot(223)  
ax4 = fig.add_subplot(224)  

gaus_lapl_1 = gaussian_laplace(img, sigma = 1, output = np.float64)
ax1.imshow(gaus_lapl_1)
ax1.set_title("Laplacian-Gaussian filtering with, sigma = 1")

gaus_lapl_2 = gaussian_laplace(img, sigma = 2, output = np.float64)
ax2.imshow(gaus_lapl_2)
ax2.set_title("Laplacian-Gaussian filtering with, sigma = 2")

gaus_lapl_4 = gaussian_laplace(img, sigma = 4, output = np.float64)
ax3.imshow(gaus_lapl_4)
ax3.set_title("Laplacian-Gaussian filtering with, sigma = 4")

gaus_lapl_8 = gaussian_laplace(img, sigma = 8, output = np.float64)
ax4.imshow(gaus_lapl_8)
ax4.set_title("Laplacian-Gaussian filtering with, sigma = 8")

plt.show()

# Canny edge detection

# First I apply Canny function from the OpenCV library
# Canny detection is a 4-stage procedure which is describe in my pdf-report
# Here it is just a single function
# Important: it uses the Sobel kernel

# I specify different thresholds to see how do they work and which one is the best
# And plot the results

edges1 = cv2.Canny(img, 0, 100)
edges2 = cv2.Canny(img, 200, 300)
edges3 = cv2.Canny(img, 100, 200)

plt.subplot(131)
plt.imshow(edges1, cmap = 'gray')
plt.title('0-100 threshold')

plt.subplot(132)
plt.imshow(edges2, cmap = 'gray')
plt.title('200-300 threshold')

plt.subplot(133)
plt.imshow(edges3, cmap = 'gray')
plt.title('100-200 threshold')

plt.show()

### Optional: Canny edge detection with Gaussian kernel

# Here I implement Canny edge detection with another library - skimage
# Because in this case I can use the Gaussian kernel
# In this case I decidet to change not thresholds, but sigmas, in order to analyze the impact of blurring and noise

from skimage import feature

edges1 = feature.canny(img, sigma = 1)
edges2 = feature.canny(img, sigma = 2)
edges3 = feature.canny(img, sigma = 4)
edges4 = feature.canny(img, sigma = 8)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols = 4, sharex=True, sharey=True)

ax1.imshow(edges1, cmap=plt.cm.gray)
ax1.set_title(r'Canny filter, $\sigma=1$', fontsize=10)

ax2.imshow(edges2, cmap=plt.cm.gray)
ax2.set_title(r'Canny filter, $\sigma=2$', fontsize=10)

ax3.imshow(edges3, cmap=plt.cm.gray)
ax3.set_title(r'Canny filter, $\sigma=4$', fontsize=10)

ax4.imshow(edges4, cmap=plt.cm.gray)
ax4.set_title(r'Canny filter, $\sigma=8$', fontsize=10)

fig.tight_layout()

plt.show()