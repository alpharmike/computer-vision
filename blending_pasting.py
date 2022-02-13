#import cv2

# cv2.namedWindow('Puppy')
#
# img1 = cv2.imread('DATA/dog_backpack.png')
# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
# img2 = cv2.imread('DATA/watermark_no_copy.png')
# img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
#
# # resizing is necessary in order to use addWeighted function
#
# img1 = cv2.resize(img1, (1200, 1200))
# img2 = cv2.resize(img2, (1200, 1200))
#
# # blended = cv2.addWeighted(src1=img1, alpha=0.5, src2=img2, beta=0.5, gamma=0)
#
# # blending an image of a smaller size to a bigger one (overlaying)
#
# img1 = cv2.resize(img1, (1200, 1200))
# img2 = cv2.resize(img2, (600, 600))
#
# large_img = img1
# small_img = img2
#
# x_offset = 0
# y_offset = 0
#
# x_end = x_offset + small_img.shape[1]
# y_end = y_offset + small_img.shape[0]
#
# large_img[y_offset:y_end, x_offset:x_end] = small_img
#
# show_img('Puppy', large_img)


