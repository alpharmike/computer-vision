import cv2
import numpy as np
from image_files import show_img

# img = cv2.imread('DATA/bricks.jpg').astype(np.float32) / 255

# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# brightening the image (gamma < 1)
# gamma = 0.5
# darkening the image (gamma > 1)
# gamma = 2
# result = np.power(img, gamma)
#
# font = cv2.FONT_HERSHEY_COMPLEX
#
# img = cv2.putText(img, 'Bricks', (20, 600), fontFace=font, fontScale=10, color=(0, 0, 255), thickness=4)

# first way of blurring

# kernel = np.ones(shape=(5, 5), dtype=np.float32) / 25

# blurred = cv2.filter2D(img, -1, kernel)

# 2nd way of blurring

# blurred2 = cv2.blur(img, ksize=(5, 5))

# 3rd way of blurring

# blurred3 = cv2.GaussianBlur(img, (5, 5), 10)

# 4th way of blurring (median blur good for reducing noise)

# blurred4 = cv2.medianBlur(img, 5)

# median blurring and bilateral filtering are appropriate for reducing noise

# new_img = cv2.imread('DATA/sammy.jpg')
#
# noise_img = cv2.imread('DATA/sammy_noise.jpg')
#
# noise_img = cv2.cvtColor(noise_img,cv2.COLOR_BGR2RGB)
#
# median = cv2.medianBlur(noise_img, 5)
#
# blur = cv2.bilateralFilter(noise_img, 9, 150, 150)
#
# show_img('Bricks', blur)
