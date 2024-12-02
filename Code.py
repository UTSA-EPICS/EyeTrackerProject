import cv2
import urllib.request
import numpy as np
import dlib
from math import hypot
import Jetson.GPIO as GPIO
import time

# Servo GPIO pin setup
servo_pin1 = 12  # GPIO pin connected to the servo (BCM numbering)
servo_pin2 = 33  # GPIO pin connected to the servo (BCM numbering)
servo_pin3 = 35  # GPIO pin connected to the servo (BCM numbering)
servo_pin4 = 37  # GPIO pin connected to the servo (BCM numbering)

min_duty_cycle = 2.5  # Min duty cycle for 0 degrees
max_duty_cycle = 12.5  # Max duty cycle for 180 degrees
servo_angle_1 = 90
servo_angle_2 = 90
servo_angle_3 = 90
servo_angle_4 = 90

#light pins
light_pin1 = 11
light_pin2 = 19
light_pin3 = 21
light_pin4 = 23

# Initialize GPIO for servo control
GPIO.setmode(GPIO.BOARD)
GPIO.setup(servo_pin1, GPIO.OUT)
GPIO.setup(servo_pin2, GPIO.OUT)
GPIO.setup(servo_pin3, GPIO.OUT)
GPIO.setup(servo_pin4, GPIO.OUT)


GPIO.setup(light_pin1, GPIO.OUT)
GPIO.setup(light_pin2, GPIO.OUT)
GPIO.setup(light_pin3, GPIO.OUT)
GPIO.setup(light_pin4, GPIO.OUT)



# Initialize GPIO for LIGHT control
GPIO.setup(light_pin1, GPIO.OUT, initial=GPIO.HIGH)
GPIO.setup(light_pin2, GPIO.OUT, initial=GPIO.HIGH)
GPIO.setup(light_pin3, GPIO.OUT, initial=GPIO.HIGH)
GPIO.setup(light_pin4, GPIO.OUT, initial=GPIO.HIGH)


def set_servo_angle1(angle):
    """Set the servo to a specific angle using software PWM."""
    duty_cycle = min_duty_cycle + (max_duty_cycle - min_duty_cycle) * (angle / 180.0)
    GPIO.output(servo_pin1, GPIO.HIGH)


    time.sleep(duty_cycle / 1000.0)  # Active high for duty cycle duration
    GPIO.output(servo_pin1, GPIO.LOW)
    time.sleep((20 - duty_cycle) / 1000.0)  # Complete 20 ms period

def set_servo_angle2(angle):
    """Set the servo to a specific angle using software PWM."""
    duty_cycle = min_duty_cycle + (max_duty_cycle - min_duty_cycle) * (angle / 180.0)

    GPIO.output(servo_pin2, GPIO.HIGH)


    time.sleep(duty_cycle / 1000.0)  # Active high for duty cycle duration
    GPIO.output(servo_pin2, GPIO.LOW)
    time.sleep((20 - duty_cycle) / 1000.0)  # Complete 20 ms period

def set_servo_angle3(angle):
    """Set the servo to a specific angle using software PWM."""
    duty_cycle = min_duty_cycle + (max_duty_cycle - min_duty_cycle) * (angle / 180.0)

    GPIO.output(servo_pin3, GPIO.HIGH)


    time.sleep(duty_cycle / 1000.0)  # Active high for duty cycle duration
    GPIO.output(servo_pin3, GPIO.LOW)
    time.sleep((20 - duty_cycle) / 1000.0)  # Complete 20 ms period

def set_servo_angle4(angle):
    """Set the servo to a specific angle using software PWM."""
    duty_cycle = min_duty_cycle + (max_duty_cycle - min_duty_cycle) * (angle / 180.0)
    GPIO.output(servo_pin4, GPIO.HIGH)

    time.sleep(duty_cycle / 1000.0)  # Active high for duty cycle duration
    GPIO.output(servo_pin4, GPIO.LOW)
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

# Function to get eye contours
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

# Function to calculate blinking ratio (EAR)
def get_blinking_ratio(eye_points, facial_landmarks):
    eye = []
    for n in eye_points:
        x = facial_landmarks.part(n).x
        y = facial_landmarks.part(n).y
        eye.append([x, y])
    eye = np.array(eye, np.int32)
    A = hypot(eye[1][0] - eye[5][0], eye[1][1] - eye[5][1])
    B = hypot(eye[2][0] - eye[4][0], eye[2][1] - eye[4][1])
    C = hypot(eye[0][0] - eye[3][0], eye[0][1] - eye[3][1])
    ear = (A + B) / (2.0 * C)
    return ear



set_servo_angle1(servo_angle_1)
set_servo_angle2(servo_angle_2)
set_servo_angle3(servo_angle_3)
set_servo_angle4(servo_angle_4)



# Main loop
try:
    while True:
        try:
            img_resp = urllib.request.urlopen(url)
            imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            img = cv2.imdecode(imgnp, cv2.IMREAD_COLOR)

            if img is None:
                print("Failed to decode image.")
                continue

            rows, cols, _ = img.shape
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img[rows - 50: rows, 0: cols] = (255, 255, 255)

            faces = detector(gray)
            for face in faces:
                landmarks = predictor(gray, face)
                left_eye, right_eye = eyes_contour_points(landmarks)

                left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
                right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
                blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

                gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
                gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
                gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2

                print("Gaze ratio:", gaze_ratio)
            if (blinking_ratio <= 0.20):
                print("Looking Down")
                GPIO.output(light_pin4, GPIO.LOW)

                GPIO.output(light_pin1, GPIO.HIGH)
                GPIO.output(light_pin2, GPIO.HIGH)
                GPIO.output(light_pin3, GPIO.HIGH)

                if (servo_angle_4 < 70):    # 90
                    servo_angle_4 += 1  # down
                    set_servo_angle4(servo_angle_4)
                    if (servo_angle_2 < 150):
                        servo_angle_2 += 1  # right
                        set_servo_angle2(servo_angle_2)
                    if (servo_angle_1 > 30):
                        servo_angle_1 -= 1  # left
                        set_servo_angle1(servo_angle_1)
                    if (servo_angle_3 < 150):
                        servo_angle_3 -= 1  # up
                        set_servo_angle3(servo_angle_3)
                        time.sleep(1)

            elif(blinking_ratio > 0.30):
                print("Looking Up")
                GPIO.output(light_pin3, GPIO.LOW)

                GPIO.output(light_pin1, GPIO.HIGH)
                GPIO.output(light_pin2, GPIO.HIGH)
                GPIO.output(light_pin4, GPIO.HIGH)

                if (servo_angle_3 < 70):    # 90
                    servo_angle_3 -= 1  # up
                    set_servo_angle3(servo_angle_3)
                    if (servo_angle_2 < 150):
                        servo_angle_2 += 1  # right
                        set_servo_angle2(servo_angle_2)
                    if (servo_angle_1 > 30):
                        servo_angle_1 -= 1  # left
                        set_servo_angle1(servo_angle_1)
                    if (servo_angle_4 < 150):
                        servo_angle_4 += 1  # down
                        set_servo_angle4(servo_angle_4)
                        time.sleep(1)

            elif gaze_ratio <= 1.3:
                print("Looking right.")
                GPIO.output(light_pin2, GPIO.LOW)

                GPIO.output(light_pin1, GPIO.HIGH)
                GPIO.output(light_pin3, GPIO.HIGH)
                GPIO.output(light_pin4, GPIO.HIGH)

                if (servo_angle_2 < 70):    # 90
                    servo_angle_2 -= 1  # right
                    set_servo_angle2(servo_angle_2)
                    if (servo_angle_1 > 30):
                        servo_angle_1 -= 1  # left
                        set_servo_angle1(servo_angle_1)
                    if (servo_angle_3 < 150):
                        servo_angle_3 += 1  # up
                        set_servo_angle3(servo_angle_3)
                    if (servo_angle_4 < 150):
                        servo_angle_4 += 1  # down
                        set_servo_angle4(servo_angle_4)
                    time.sleep(1)


                    set_servo_angle2(servo_angle_2)
            elif (gaze_ratio > 1.3):
                print("Looking left.")
                GPIO.output(light_pin1, GPIO.LOW)

                GPIO.output(light_pin2, GPIO.HIGH)
                GPIO.output(light_pin3, GPIO.HIGH)
                GPIO.output(light_pin4, GPIO.HIGH)

                if (servo_angle_1 < 70):    # 90
                    servo_angle_1 += 1  # left
                    set_servo_angle1(servo_angle_1)
                    if (servo_angle_2 < 150):
                        servo_angle_2 += 1  # right
                        set_servo_angle2(servo_angle_2)
                    if (servo_angle_3 < 150):
                        servo_angle_3 += 1  # up
                        set_servo_angle3(servo_angle_3)
                    if (servo_angle_4 < 150):
                        servo_angle_4 += 1  # down
                        set_servo_angle4(servo_angle_4)
                        time.sleep(1)

            cv2.imshow('live Cam Testing', img)
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
