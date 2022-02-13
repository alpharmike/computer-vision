import cv2
import matplotlib.pyplot as plt
import numpy as np
from image_files import show_img

face_cascade = cv2.CascadeClassifier(
    'D:/PycharmProjects/ImageProcessing/DATA/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
    'D:/PycharmProjects/ImageProcessing/DATA/haarcascades/haarcascade_eye.xml')
plate_cascade = cv2.CascadeClassifier(
    'D:/PycharmProjects/ImageProcessing/DATA/haarcascades/haarcascade_russian_plate_number.xml')

smile_cascade = cv2.CascadeClassifier(
    'D:/PycharmProjects/ImageProcessing/DATA/haarcascades/haarcascade_smile.xml')


def detect_face(image):
    face_image = image.copy()
    face_rects = face_cascade.detectMultiScale(face_image, scaleFactor=1.5, minNeighbors=5)
    # eye_rects = eye_cascade.detectMultiScale(face_image, scaleFactor=2, minNeighbors=10)
    smile_rects = smile_cascade.detectMultiScale(face_image, scaleFactor=2, minNeighbors=10)
    for (x, y, w, h) in face_rects:
        cv2.rectangle(face_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # for (x, y, w, h) in eye_rects:
    #     cv2.rectangle(face_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    for (x, y, w, h) in smile_rects:
        cv2.rectangle(face_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return face_image


def detect_eyes(image):
    face_image = image.copy()
    eye_rects = eye_cascade.detectMultiScale(face_image, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in eye_rects:
        cv2.rectangle(face_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return face_image


def detect_plate(image):
    plate_image = image.copy()
    plate_rects = plate_cascade.detectMultiScale(plate_image, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in plate_rects:
        cv2.rectangle(plate_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return plate_image


def detect_and_blur_plate(image):
    plate_image = image.copy()
    roi = image.copy()
    plate_rects = plate_cascade.detectMultiScale(plate_image, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in plate_rects:
        # cv2.rectangle(plate_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi = roi[y:y+h][x:x+w]
        blurred_roi = cv2.medianBlur(src=roi, ksize=15)
        plate_image[y:y+h][x:x+w] = blurred_roi

    return plate_image


# car_plate = cv2.imread('DATA/car_plate.jpg')
# plate_detection = detect_and_blur_plate(car_plate)
# print(plate_detection)
# nadia = cv2.imread('DATA/Nadia_Murad.jpg')
# face_rec = detect_eyes(nadia)
# show_img('plate', plate_detection)