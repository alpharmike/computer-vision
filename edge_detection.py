import cv2
import numpy as np
import matplotlib.pyplot as plt
from image_files import show_img


def blur_image_manual(image):
    kernel = np.ones(shape=(5, 5), dtype=np.float32) / 25
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred


def edge_detector(image):
    median_val = np.median(image)
    lower = int(max(0, 0.7 * median_val))
    upper = int(min(255, 1.3 * median_val))
    edges = cv2.Canny(image, threshold1=lower, threshold2=upper)
    return edges


img = cv2.imread('DATA/sammy_face.jpg')
blurred_img = blur_image_manual(img)
blurred_img2 = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=10)
blurred_img3 = cv2.bilateralFilter(img, 9, 150, 150)
blurred_img4 = cv2.blur(img, ksize=(9, 9))
edges_recognized = edge_detector(blurred_img4)
# show_img('Edges', edges_recognized)
