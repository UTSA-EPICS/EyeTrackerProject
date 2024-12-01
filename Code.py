import cv2
import urllib.request
import numpy as np
import dlib
from math import hypot
import Jetson.GPIO as GPIO
import time


# Servo GPIO pin setup
servo_pin = 18  # GPIO pin connected to the servo (BCM numbering)
min_duty_cycle = 2.5  # Min duty cycle for 0 degrees
max_duty_cycle = 12.5  # Max duty cycle for 180 degrees


# Initialize GPIO for servo control
GPIO.setmode(GPIO.BCM)
GPIO.setup(servo_pin, GPIO.OUT)


def set_servo_angle(angle):
   """Set the servo to a specific angle using software PWM."""
   duty_cycle = min_duty_cycle + (max_duty_cycle - min_duty_cycle) * (angle / 180.0)
   GPIO.output(servo_pin, GPIO.HIGH)
   time.sleep(duty_cycle / 1000.0)  # Active high for duty cycle duration
   GPIO.output(servo_pin, GPIO.LOW)
   time.sleep((20 - duty_cycle) / 1000.0)  # Complete 20 ms period


# Initialize variables
keyboard = np.zeros((600, 1000, 3), np.uint8)
url = 'http://192.168.1.171/cam-lo.jpg'
cv2.namedWindow("live Cam Testing", cv2.WINDOW_AUTOSIZE)


# Create VideoCapture object (though not used directly here)
cap = cv2.VideoCapture(url)


# Initialize facial feature detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def midpoint(p1, p2):
   return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


def get_gaze_ratio(eye_points, facial_landmarks):
   left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                               (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                               (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                               (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                               (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                               (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)],
                              np.int32)


   height, width, _ = img.shape
   mask = np.zeros((height, width), np.uint8)
   cv2.polylines(mask, [left_eye_region], True, 255, 2)
   cv2.fillPoly(mask, [left_eye_region], 255)
   eye = cv2.bitwise_and(gray, gray, mask=mask)


   min_x = np.min(left_eye_region[:, 0])
   max_x = np.max(left_eye_region[:, 0])
   min_y = np.min(left_eye_region[:, 1])
   max_y = np.max(left_eye_region[:, 1])


   gray_eye = eye[min_y: max_y, min_x: max_x]
   _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
   height, width = threshold_eye.shape
   left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
   left_side_white = cv2.countNonZero(left_side_threshold)


   right_side_threshold = threshold_eye[0: height, int(width / 2): width]
   right_side_white = cv2.countNonZero(right_side_threshold)


   if left_side_white == 0:
       gaze_ratio = 1
   elif right_side_white == 0:
       gaze_ratio = 5
   else:
       gaze_ratio = left_side_white / right_side_white
   return gaze_ratio


def eyes_contour_points(facial_landmarks):
   left_eye = []
   right_eye = []
   for n in range(36, 42):  # Left eye landmark points
       x = facial_landmarks.part(n).x
       y = facial_landmarks.part(n).y
       left_eye.append([x, y])
   for n in range(42, 48):  # Right eye landmark points
       x = facial_landmarks.part(n).x
       y = facial_landmarks.part(n).y
       right_eye.append([x, y])
   left_eye = np.array(left_eye, np.int32)
   right_eye = np.array(right_eye, np.int32)
   return left_eye, right_eye


angle = 0
set_servo_angle(angle)
time.sleep(2)
angle = 90
set_servo_angle(angle)


# Main loop
try:
   while True:
       try:
           # Fetch the image from the URL
           img_resp = urllib.request.urlopen(url)
           imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)


           # Decode the image into an OpenCV image
           img = cv2.imdecode(imgnp, cv2.IMREAD_COLOR)


           # Convert to grayscale for face detection
           gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


           # Detect faces in the grayscale image
           faces = detector(gray)
           for face in faces:
               landmarks = predictor(gray, face)
               left_eye, right_eye = eyes_contour_points(landmarks)


               # Detect gaze ratio
               gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
               gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
               gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2


               print("Gaze ratio:", gaze_ratio)


               # Perform actions based on gaze ratio
               if gaze_ratio <= 1.3:  # Looking right
                   print("Looking right.")
                   angle = 0
                   set_servo_angle(angle)  # Move servo clockwise
                   print(angle)
                   time.sleep(2)


               else:  # Looking left
                   print("Looking left.")
                   angle = 90
                   set_servo_angle(angle)  # Move servo counterclockwise
                   print(angle)
                   time.sleep(2)




           # Display the frame with annotations
           cv2.imshow('live Cam Testing', img)


           # Wait for 'q' key press to quit
           key = cv2.waitKey(5)
           if key == ord('q'):
               break


       except Exception as e:
           print(f"Error processing frame: {e}")
           continue


finally:
   GPIO.cleanup()
   cv2.destroyAllWindows()
   print("GPIO cleaned up.")
