import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(
    'D:\\Computer Vision\\1. Course Overview and Introduction\\Computer-Vision-with-Python\\DATA\\dog_backpack.png')

cv2.imshow('Puppy', img)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

vertices = np.array([[250, 700], [425, 400], [650, 700]])

pts = vertices.reshape((-1, 1, 2))

cv2.polylines(img_rgb, [pts], isClosed=True, color=(255, 0, 0))
