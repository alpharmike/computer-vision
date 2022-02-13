import cv2
import numpy as np

# img = np.zeros((512, 512, 3))
#
# cv2.namedWindow('Drawing')
#
# drawing = False
# initial_x = -1
# initial_y = -1


def show_img(window_name, image):
    while True:
        cv2.imshow(window_name, image)
        if cv2.waitKey() & 0xFF == 27:
            break
    cv2.destroyAllWindows()



# def draw_circle(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         cv2.circle(img, (x, y), 100, (255, 0, 0), -1)
#     elif event == cv2.EVENT_RBUTTONDOWN:
#         cv2.circle(img, (x, y), 100, (0, 255, 0), -1)
#
#
# def draw_rectangle(event, x, y, flags, params):
#
#     global initial_x, initial_y, drawing
#
#     if event == cv2.EVENT_LBUTTONDOWN:
#         drawing = True
#         initial_x = x
#         initial_y = y
#     elif event == cv2.EVENT_MOUSEMOVE:
#         if drawing == True:
#             cv2.rectangle(img, (initial_x, initial_y), (x, y), (255, 0, 0), -1)
#     elif event == cv2.EVENT_LBUTTONUP:
#         drawing = False
#         cv2.rectangle(img, (initial_x, initial_y), (x, y), (255, 0, 0), -1)
#
#
# cv2.setMouseCallback('Drawing', draw_rectangle)
#
# while True:
#     cv2.imshow('Drawing', img)
#
#     if cv2.waitKey(20) & 0xFF == 27:
#         break
#
# cv2.destroyAllWindows()
