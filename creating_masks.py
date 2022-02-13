import cv2
import numpy as np
from image_files import show_img

img1 = cv2.imread('DATA/dog_backpack.png')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.imread('DATA/watermark_no_copy.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

img2 = cv2.resize(img2, (600, 600))

x_offset = img1.shape[1] - 600
y_offset = img1.shape[0] - 600

roi = img1[y_offset:1401, x_offset:934]

img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

mask_inv = cv2.bitwise_not(img2gray)

white_bg = np.full(img2.shape, 255, dtype=np.uint8)

new_bg = cv2.bitwise_or(white_bg, white_bg, mask=mask_inv)

fore_ground = cv2.bitwise_or(img2, img2, mask=mask_inv)

final_roi = cv2.bitwise_or(roi, fore_ground)

large_img = img1

small_img = final_roi

large_img[y_offset:y_offset + small_img.shape[0], x_offset:x_offset + small_img.shape[1]] = small_img

show_img('Image', large_img)
