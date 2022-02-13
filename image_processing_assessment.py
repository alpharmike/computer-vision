import cv2
import numpy as np
from image_files import show_img

img = cv2.imread("DATA/giraffes2.jpg")

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

ret, thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)

kernel = np.ones((4, 4), np.float32) / 10

# kernel = np.multiply(kernel, 0.1)

blurred = cv2.filter2D(img, -1, kernel)

kernel2 = np.ones((5, 5), np.uint8)

sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=5)

sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=5)

gradient = cv2.morphologyEx(gray_img, cv2.MORPH_GRADIENT, kernel2)

show_img("Giraffes", gradient)
