# EPICS ALS Eye Tracker

## 🔍 Overview  
This branch is dedicated to **Eye Tracker Development**.

**Disclaimer:** This software is intended for research/educational use. It does not contain or process personal medical data.

---

## 👁️ Main Eye Tracker Setup Instructions

To configure the video stream device in your Python eye tracker script:

1. Locate **line 16** in your script.
2. Set the camera source using `cv2.VideoCapture(#)`:
   - `cv2.VideoCapture(0)` uses the **default video device**, typically your laptop's built-in webcam.
   - To select a different camera, replace `0` with the appropriate device index:
     - For example, if your laptop camera is broken and you are using your **phone as a virtual webcam**, it might show up as the second video device — use `cv2.VideoCapture(1)`.

💡 *Tip:* You can test different numbers (`0`, `1`, `2`, etc.) to see which one corresponds to your desired video input.

---
