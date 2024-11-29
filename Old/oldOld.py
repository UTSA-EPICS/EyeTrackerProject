from unittest.mock import right

import cv2
import numpy as np
import dlib
from math import hypot
from Tools.scripts.generate_global_objects import Printer
from numpy.ma.core import make_mask
from pyscreeze import center

# (0) is for the first index for the webcam, to pick a different webcam increase the number
cap = cv2.VideoCapture(1)
# object to detect faces
detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

def get_blinking_ratio(eye_points, facial_landmarks):
    # get the left point and right point of right eye
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    # finding the mid-point of 37 and 38
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    # finding the mid-point of 41 and 40
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    # draws a horizontal line across the right eye (from point 36 to 37)
    #hori_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    # draws a vertical line from the center of the right eye
    #ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

    # calculates the lenght
    hori_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hori_line_length / ver_line_lenght

    return ratio



font =cv2.FONT_HERSHEY_PLAIN

# creates a window called "frame" to display the video capture
while True:
    _, frame = cap.read()
    # converting capture to greyscale for better video computing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # array where we have all the faces
    faces = detector(gray)
    for face in faces:
        # gets the landmarks of the face
        landmarks = predictor(gray, face)
        # Detects Blincking
        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        if blinking_ratio > 5.7:
            cv2.putText(frame, "Blinking", (50, 150), font, 7, (255, 0, 0))

        # Gaze detections
        left_eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                    (landmarks.part(37).x, landmarks.part(37).y),
                                    (landmarks.part(38).x, landmarks.part(38).y),
                                    (landmarks.part(39).x, landmarks.part(39).y),
                                    (landmarks.part(40).x, landmarks.part(40).y),
                                    (landmarks.part(41).x, landmarks.part(41).y)], np.int32)

        # cv2.polylines(frame, [left_eye_region], True, (0, 0, 255), 2)

        # grabs ratio of camera
        height, width, _ = frame.shape
        # making a mask for the eye
        mask = np.zeros((height, width), np.uint8)
        # filling in the 'Mask' window with the mask of the "left_eye_region"
        cv2.polylines(mask, [left_eye_region], True, 255, 2)
        cv2.fillPoly(mask, [left_eye_region], 255)
        left_eye = cv2.bitwise_and(gray, gray, mask = mask)


        # extreme points of the eye
        min_x = np.min(left_eye_region[:, 0])
        max_x = np.max(left_eye_region[:, 0])
        min_y = np.min(left_eye_region[:, 1])
        max_y = np.max(left_eye_region[:, 1])

        gray_eye = left_eye[min_y: max_y, min_x: max_x]
        _, threshhold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY_INV)

        eye = cv2.resize(gray_eye, None, fx = 5, fy = 5)
        cv2.imshow("Eye", eye)
        threshhold_eye = cv2.resize(threshhold_eye, None, fx = 5, fy = 5)
        cv2.imshow("Threshold", threshhold_eye)
        cv2.imshow("Left_eye", left_eye)



    cv2.imshow('Frame', frame)

    key = cv2.waitKey(1)
    # if you press the "esc" key it stoped the program
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
