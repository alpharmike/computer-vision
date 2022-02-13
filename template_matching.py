import cv2
import numpy as np
import matplotlib.pyplot as plt
from image_files import show_img

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF',
           'cv2.TM_SQDIFF_NORMED']

sammy = cv2.imread('DATA/sammy.jpg')

sammy_face = cv2.imread('DATA/sammy_face.jpg')

for choice in methods:
    img_copy = sammy.copy()

    method = eval(choice)

    res = cv2.matchTemplate(img_copy, sammy_face, method)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    if method in [cv2.TM_SQDIFF, cv2.TM_CCOEFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc

    height, width, channels = sammy_face.shape
    bottom_right = (top_left[0] + width, top_left[1] + height)
    cv2.rectangle(img_copy, top_left, bottom_right, color=(0, 0, 255), thickness=10)

    plt.subplot(121)
    plt.imshow(res)
    plt.title('Heat Map of Template Matching')

    plt.subplot(122)
    plt.imshow(img_copy)
    plt.title('Detection of Template')
    plt.suptitle(choice)

    plt.show()

    print('\n')
    print('\n')

# my_method = eval('cv2.TM_CCOEFF')
#
# my_res = cv2.matchTemplate(sammy, sammy_face, cv2.TM_CCOEFF_NORMED)

# show_img('Window', my_res)

# show_img('Template', res)
