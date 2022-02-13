import cv2
import numpy as np
from image_files import show_img

img = cv2.imread("DATA/sudoku.jpg", 0)

sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)

sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

blended = cv2.addWeighted(src1=sobel_x, alpha=0.5, src2=sobel_y, beta=0.5, gamma=0)

ret, thresh = cv2.threshold(src=blended, thresh=127, maxval=255, type=cv2.THRESH_BINARY_INV)

kernel = np.ones((4, 4), np.uint8)

gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

show_img("Sudoku", gradient)
