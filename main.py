import cv2
import numpy as np
from scipy import stats

# load image
img = cv2.imread('Image/balon.jpg')

# convert image to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# calculate mean, standard deviation, and variance
mean, std_dev = cv2.meanStdDev(hsv)
variance = np.square(std_dev)

# calculate mode
mode = stats.mode(hsv.reshape(-1,3), axis=0)[0][0]

# calculate skewness
skewness = stats.skew(hsv.reshape(-1,3), axis=0)

# print results
print('Mean:', mean.flatten())
print('Standard Deviation:', std_dev.flatten())
print('Variance:', variance.flatten())
print('Mode:', mode)
print('Skewness:', skewness.flatten())