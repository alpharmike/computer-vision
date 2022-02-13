import cv2
import numpy as np
import matplotlib.pyplot as plt
from image_files import show_img
from contour_detection import find_internal_contours, find_external_contours

coins = cv2.imread('DATA/coins.jpg')
blurred_coins = cv2.medianBlur(coins, 11)
gray_coins = cv2.cvtColor(blurred_coins, cv2.COLOR_BGR2GRAY)
ret , thresh = cv2.threshold(gray_coins, 160, 255, cv2.THRESH_BINARY_INV)
internal = find_internal_contours(thresh)
external = find_external_contours(thresh)
show_img('Coins', external)