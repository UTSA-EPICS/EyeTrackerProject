import cv2
import numpy as np
import dlib
from math import hypot
import time

# Initialize variables
keyboard = np.zeros((600, 1000, 3), np.uint8)
cv2.namedWindow("live Cam Testing", cv2.WINDOW_AUTOSIZE)

# Counters
frames = 0
frames_to_blink = 6

# Create VideoCapture object
cap = cv2.VideoCapture(1)  # Use 0 if 1 does not work
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Initialize facial feature detector
detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
except RuntimeError as e:
    print("Error: Could not load shape predictor. Ensure the file is in the correct directory.")
    exit()

def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

def get_gaze_ratio(eye_points, facial_landmarks, frame, gray):
    left_eye_region = np.array([(facial_landmarks.part(point).x, facial_landmarks.part(point).y) for point in eye_points], np.int32)
    
    mask = np.zeros(gray.shape, np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)
    
    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])
    
    gray_eye = eye[min_y: max_y, min_x: max_x]
    if gray_eye.size == 0:
        return 1
    
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    left_side_white = cv2.countNonZero(threshold_eye[:, :width // 2])
    right_side_white = cv2.countNonZero(threshold_eye[:, width // 2:])
    
    if left_side_white == 0:
        return 1
    elif right_side_white == 0:
        return 5
    return left_side_white / right_side_white

def eyes_contour_points(facial_landmarks):
    left_eye = np.array([[facial_landmarks.part(n).x, facial_landmarks.part(n).y] for n in range(36, 42)], np.int32)
    right_eye = np.array([[facial_landmarks.part(n).x, facial_landmarks.part(n).y] for n in range(42, 48)], np.int32)
    return left_eye, right_eye

def get_blinking_ratio(eye_points, facial_landmarks):
    eye = np.array([[facial_landmarks.part(n).x, facial_landmarks.part(n).y] for n in eye_points], np.int32)
    A = hypot(eye[1][0] - eye[5][0], eye[1][1] - eye[5][1])
    B = hypot(eye[2][0] - eye[4][0], eye[2][1] - eye[4][1])
    C = hypot(eye[0][0] - eye[3][0], eye[0][1] - eye[3][1])
    return (A + B) / (2.0 * C)

# Main loop
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        frame = cv2.resize(frame, None, fx=0.3, fy=0.3)
        keyboard[:] = (26, 26, 26)
        frames += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        for face in faces:
            landmarks = predictor(gray, face)
            left_eye, right_eye = eyes_contour_points(landmarks)
            
            left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
            right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
            blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
            
            cv2.polylines(frame, [left_eye], True, (0, 0, 255), 2)
            cv2.polylines(frame, [right_eye], True, (0, 0, 255), 2)
            
            gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks, frame, gray)
            gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks, frame, gray)
            gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2
            
            #print("Gaze ratio:", gaze_ratio)
            #print("Blink ratio:", blinking_ratio)
            
            if blinking_ratio <= 0.20:
                print("Looking Down")
            elif blinking_ratio > 0.25:
                print("Looking Up")
            elif gaze_ratio <= 1.3:
                print("Looking right.")
            else:
                print("Looking left.")
                           
        cv2.imshow('live Cam Testing', frame)
        key = cv2.waitKey(5)
        if key == ord('q'):
            break

except Exception as e:
    print(f"Error: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()
