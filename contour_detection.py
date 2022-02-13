import cv2
import numpy as np
import matplotlib.pyplot as plt
from image_files import show_img


def find_external_contours(img):
    image, contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if len(image.shape) == 2:
        external_contours = np.zeros(image.shape)
    else:
        width, height = image.shape[0], image.shape[1]
        external_contours = np.zeros((width, height))
    for i in range(len(contours)):
        if hierarchy[0][i][3] == -1:
            cv2.drawContours(external_contours, contours, i, 255, -1)
    return external_contours


def find_internal_contours(img):
    image, contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if len(image.shape) == 2:
        internal_contours = np.zeros(image.shape)
    else:
        width, height = image.shape[0], image.shape[1]
        internal_contours = np.zeros((width, height))
    for i in range(len(contours)):
        if hierarchy[0][i][3] != -1:
            cv2.drawContours(internal_contours, contours, i, 255, -1)
    return internal_contours


target = cv2.imread('DATA/internal_external.png', 0)
internal = find_internal_contours(target)
external = find_external_contours(target)

# show_img('Internal Contours', internal)
# show_img('External Contours', external)
