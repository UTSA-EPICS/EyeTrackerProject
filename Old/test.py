import cv2
import numpy as np
import dlib
import pyglet
import time
import urllib.request

# Load sound for blink
sound = pyglet.media.load("../Sounds/blink_FNZ3zVv.mp3", streaming=False)

# IP camera URL
ip_camera_url = "http://192.168.86.212/cam-hi.jpg"

# Dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Keyboard settings
keyboard = np.zeros((600, 1000, 3), np.uint8)
keys_set_1 = {
    0: "Q", 1: "W", 2: "E", 3: "R", 4: "T", 5: "A",
    6: "S", 7: "D", 8: "F", 9: "G", 10: "<",
}


def draw_letters(letter_index, text, letter_light):
    """Draw a single key on the keyboard."""
    x = (letter_index % 5) * 200
    y = (letter_index // 5) * 200
    width, height = 200, 200
    th = 3  # thickness of the rectangle

    # Light key background if selected
    if letter_light:
        cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (255, 255, 255), -1)
    else:
        cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (0, 0, 0), -1)

    # Draw key borders and text
    cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (255, 0, 0), th)
    font_letter = cv2.FONT_HERSHEY_PLAIN
    font_scale = 10
    font_thickness = 4
    text_size = cv2.getTextSize(text, font_letter, font_scale, font_thickness)[0]
    text_x = x + (width - text_size[0]) // 2
    text_y = y + (height + text_size[1]) // 2
    cv2.putText(keyboard, text, (text_x, text_y), font_letter, font_scale, (255, 0, 0), font_thickness)


def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


def get_blinking_ratio(eye_points, facial_landmarks):
    """Calculate blinking ratio for a given eye."""
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    # Euclidean distances
    hor_line_length = np.linalg.norm(np.array(left_point) - np.array(right_point))
    ver_line_length = np.linalg.norm(np.array(center_top) - np.array(center_bottom))

    return hor_line_length / ver_line_length


def fetch_frame_from_ip_camera(url):
    """Fetch frame from an IP camera."""
    try:
        response = urllib.request.urlopen(url)
        img_array = np.array(bytearray(response.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        print(f"Error fetching frame: {e}")
        return None


# Main variables
frames = 0
blinking_frames = 0
letter_index = 0
frames_to_blink = 6
frames_active_letter = 9
text = ""

# Create keyboard layout
for i, letter in keys_set_1.items():
    draw_letters(i, letter, False)

while True:
    # Fetch and process frame from IP camera
    frame = fetch_frame_from_ip_camera(ip_camera_url)
    if frame is None:
        print("Failed to capture frame. Retrying...")
        time.sleep(0.1)
        continue

    # Resize and convert to grayscale
    frame = cv2.resize(frame, None, fx=0.3, fy=0.3)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray)
    if len(faces) == 0:
        print("No face detected!")
        continue

    # Process the first detected face
    for face in faces:
        landmarks = predictor(gray, face)

        # Calculate blinking ratios for both eyes
        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)

        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
        print(f"Blinking Ratio: {blinking_ratio}")

        if blinking_ratio > 5.0: #5.7
            blinking_frames += 1
            if blinking_frames == frames_to_blink:
                sound.play()
                selected_letter = keys_set_1[letter_index]
                text += selected_letter
                print(f"Letter Selected: {selected_letter}")
        else:
            blinking_frames = 0

        # Move to the next letter
        frames += 1
        if frames == frames_active_letter:
            frames = 0
            letter_index = (letter_index + 1) % len(keys_set_1)

    # Update keyboard display
    keyboard = np.zeros((600, 1000, 3), np.uint8)
    for i, letter in keys_set_1.items():
        draw_letters(i, letter, i == letter_index)

    # Display the output
    cv2.imshow("Frame", frame)
    cv2.imshow("Keyboard", keyboard)

    # Display the typed text
    board = np.zeros((200, 1000), np.uint8)
    board[:] = 255
    cv2.putText(board, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    cv2.imshow("Board", board)

    # Exit on ESC key
    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()
