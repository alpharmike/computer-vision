import cv2
from image_files import show_img

img = cv2.imread('DATA/rainbow.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

min_val, threshold = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
min_val, threshold1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
min_val, threshold2 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
min_val, threshold3 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
min_val, threshold4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

img2 = cv2.imread('DATA/crossword.jpg')

img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

min_val2, threshold5 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)

adapt_thresh = cv2.adaptiveThreshold(img2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 8)

blended = cv2.addWeighted(src1=img2, alpha=0.6, src2=adapt_thresh, beta=0.4, gamma=0)

show_img('Crossword', blended)
