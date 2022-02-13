import cv2
import time
from edge_detection import *
from edge_detection import edge_detector
from corner_detection import harris_corner_detector, shi_tomasi_corner_detector
from detection import detect_face, detect_eyes

top_left_clicked = False
bottom_right_clicked = False
pt1 = (0, 0)
pt2 = (0, 0)


def draw_rect(event, x, y, flags, params):
    global top_left_clicked, bottom_right_clicked, pt1, pt2
    if event == cv2.EVENT_LBUTTONDOWN:
        if top_left_clicked and bottom_right_clicked:
            pt1 = (0, 0)
            pt2 = (0, 0)
            top_left_clicked = False
            bottom_right_clicked = False
        if top_left_clicked:
            pt1 = (x, y)
            top_left_clicked = True
        elif bottom_right_clicked == False:
            pt2 = (x, y)
            bottom_right_clicked = True


def video_capture(window_name):
    cap = cv2.VideoCapture(0)
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_rect)
    while True:
        ret, frame = cap.read()
        # frame = edge_detector(frame)
        # frame = harris_corner_detector(frame)
        # frame = detect_face(frame)
        frame1 = edge_detector(frame)
        frame2 = detect_face(frame)
        if top_left_clicked:
            cv2.circle(frame, center=pt1, radius=5, color=(0, 255, 0), thickness=-1)
        if top_left_clicked and bottom_right_clicked:
            cv2.rectangle(frame, pt1, pt2, color=(0, 0, 255), thickness=3)
        cv2.imshow(window_name, frame1)
        cv2.imshow('face detection', frame2)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


def load_video(path):
    cap = cv2.VideoCapture(path)
    if cap.isOpened() == False:
        print('Problem Opening The File!')
    FPS = cap.get(cv2.CAP_PROP_FPS)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            time.sleep(1 / FPS)
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


# load_video('DATA/video_capture.mp4')
video_capture('Frame')