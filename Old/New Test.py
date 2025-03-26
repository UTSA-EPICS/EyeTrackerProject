import cv2
import urllib.request
import numpy as np

import dlib
from math import hypot

# font for all the text (not sure if needed)
font = cv2.FONT_HERSHEY_PLAIN

keyboard = np.zeros((600, 1000, 3), np.uint8)


# Replace the URL with the IP camera's stream URL
url = 'http://192.168.86.212/cam-hi.jpg'
cv2.namedWindow("live Cam Testing", cv2.WINDOW_AUTOSIZE)

# Create a VideoCapture object
cap = cv2.VideoCapture(url)
# Creates an empty screen for direction board
board = np.zeros((300, 1400), np.uint8)
board[:] = 255

# Detects facial features
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def draw_menu():
    rows, cols, _ = keyboard.shape
    th_lines = 4 # thickness lines
    cv2.line(keyboard, (int(cols/2) - int(th_lines), 0),(int(cols/2) - int(th_lines/2), rows),
             (51, 51, 51), th_lines)
    cv2.putText(keyboard, "LEFT", (80, 300), font, 6, (255, 255, 255), 5)
    cv2.putText(keyboard, "RIGHT", (80 + int(cols/2), 300), font, 6, (255, 255, 255), 5)
    cv2.putText(keyboard, "UP", (430, 150), font, 6, (255, 255, 255), 5)
    cv2.putText(keyboard, "Down", (380, 500), font, 6, (255, 255, 255), 5)

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

def get_gaze_ratio(eye_points, facial_landmarks):
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)
    # cv2.polylines(frame, [left_eye_region], True, (0, 0, 255), 2)

    height, width, _ = img_resp.shape
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
    for n in range(36, 42):
        x = facial_landmarks.part(n).x
        y = facial_landmarks.part(n).y
        left_eye.append([x, y])
    for n in range(42, 48):
        x = facial_landmarks.part(n).x
        y = facial_landmarks.part(n).y
        right_eye.append([x, y])
    left_eye = np.array(left_eye, np.int32)
    right_eye = np.array(right_eye, np.int32)
    return left_eye, right_eye

def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    #hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    #ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_lenght / ver_line_lenght
    return ratio


frames = 0
letter_index = 0
blinking_frames = 0
# how many frames the eyes have to be closed to select the letter
frames_to_blink = 6
# how fast the letters switch
frames_active_letter = 9

# Text and keyboard settings
text = ""
keyboard_selected = "left"
last_keyboard_selected = "left"
select_keyboard_menu = True
keyboard_selection_frames = 0





# Check if the IP camera stream is opened successfully
if not cap.isOpened():
    print("Failed to open the IP camera stream")
    exit()

while True:
    try:
        # Fetch the frame from the URL (HTTPResponse object)
        img_resp = urllib.request.urlopen(url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)

        # Decode the image into OpenCV-compatible format
        im = cv2.imdecode(imgnp, cv2.IMREAD_COLOR)

        # Ensure the decoded image is valid
        if im is None:
            print("Failed to decode image.")
            continue

        # Now you can safely access the shape of the decoded image 'im'
        rows, cols, _ = im.shape  # Get the dimensions of the image

        # Convert the image to grayscale for face detection
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        # Optionally draw a white space for the loading bar at the bottom of the frame
        im[rows - 50: rows, 0: cols] = (255, 255, 255)

        # Optionally show a menu or keyboard
        if select_keyboard_menu:
            draw_menu()

        # Face detection using dlib detector
        faces = detector(gray)
        for face in faces:
            landmarks = predictor(gray, face)
            left_eye, right_eye = eyes_contour_points(landmarks)

            # Detect blinking ratio for both eyes
            left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
            right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
            blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

            # Draw contours around the eyes
            cv2.polylines(im, [left_eye], True, (0, 0, 255), 2)
            cv2.polylines(im, [right_eye], True, (0, 0, 255), 2)

            # If selecting from the keyboard menu, detect gaze ratio to select left or right
            if select_keyboard_menu:
                gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
                gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
                gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2

                print("Gaze ratio:", gaze_ratio)

                if gaze_ratio <= 1.3:  # If looking towards the right side
                    keyboard_selected = "right"
                    keyboard_selection_frames += 1
                    if keyboard_selection_frames == 30:
                        select_keyboard_menu = False
                        frames = 0
                        keyboard_selection_frames = 0
                else:  # If looking towards the left side
                    keyboard_selected = "left"
                    keyboard_selection_frames += 1
                    if keyboard_selection_frames == 30:
                        select_keyboard_menu = False
                        frames = 0
                        keyboard_selection_frames = 0

            else:
                # Detect the blinking action to select the key on the keyboard
                if blinking_ratio > 4.0:
                    blinking_frames += 1
                    frames -= 1
                    cv2.polylines(im, [left_eye], True, (0, 255, 0), 2)
                    cv2.polylines(im, [right_eye], True, (0, 255, 0), 2)

                    # Blinking loading bar
                    percentage_blinking = blinking_frames / frames_to_blink
                    loading_x = int(cols * percentage_blinking)
                    cv2.rectangle(im, (0, rows - 50), (loading_x, rows), (51, 51, 51), -1)

        # Display the frame with all annotations
        cv2.imshow('live Cam Testing', im)

        # Check for the 'q' key to exit the loop
        key = cv2.waitKey(5)
        if key == ord('q'):
            break

    except Exception as e:
        print(f"Error fetching or decoding frame: {e}")
        continue

# Release resources after the loop ends
cap.release()
cv2.destroyAllWindows()
