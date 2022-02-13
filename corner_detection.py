import cv2
import numpy as np
import matplotlib.pyplot as plt
from image_files import show_img


def harris_corner_detector(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_arr = np.float32(gray_img)
    destination = cv2.cornerHarris(gray_arr, blockSize=2, ksize=3, k=0.04)
    destination = cv2.dilate(destination, None)
    img[destination > 0.01 * destination.max()] = [255, 0, 0]
    return img


def shi_tomasi_corner_detector(img, max_corners):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    possible_corners = cv2.goodFeaturesToTrack(gray_img, max_corners, 0.01, 10)
    possible_corners = np.int0(possible_corners)
    for choice in possible_corners:
        x_pos, y_pos = choice.ravel()
        cv2.circle(img, (x_pos, y_pos), 3, color=(255, 0, 0), thickness=-1)
    return img


flat_chess = cv2.imread('DATA/flat_chessboard.png')
flat_chess_normal = cv2.cvtColor(flat_chess, cv2.COLOR_BGR2RGB)
flat_chess_gray = cv2.cvtColor(flat_chess, cv2.COLOR_BGR2GRAY)
real_chess = cv2.imread('DATA/real_chessboard.jpg')
real_chess_normal = cv2.cvtColor(real_chess, cv2.COLOR_BGR2RGB)
real_chess_gray = cv2.cvtColor(real_chess, cv2.COLOR_BGR2GRAY)

target = harris_corner_detector(flat_chess_normal)
target2 = shi_tomasi_corner_detector(real_chess_normal, 150)

# plt.imshow(target2, cmap='gray')
#
# plt.show()
