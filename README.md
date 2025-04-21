# EPICS ALS Eye Tracker

## 🔍 Overview  
This branch is for the **Eye Tracker Demonstration**, which uses:  
- An **ESP32-CAM** for remote video streaming  
- An **Arduino** for controlling **servo motors** and **LED indicators**

**Disclaimer:** This software is intended for research/educational use. It does not contain or process personal medical data.

---

## 📷 ESP32 Setup Instructions

1. Open the ESP32-CAM code:  
   `Esp32_camera_WORKING_with_python`
2. Modify the Wi-Fi credentials in lines **5** and **6**:
   - Ensure you are connected to a **2.4GHz Wi-Fi network** (You can set your phone hotspot to 2.4GHz).
   - Line 5 → Set your **Wi-Fi SSID** (network name)  
   - Line 6 → Set your **Wi-Fi password**
3. Flash the code onto the ESP32.
4. After flashing, copy the **local IP address** shown in the serial monitor — this will be the URL for the video stream.

---

## 🔧 Arduino Setup Instructions

1. Open the Arduino sketch:  
   `Eye_Tracking_Arduino_V2.0`
2. Upload the code to the Arduino.

---

## 🔌 Pin Setup

### 🎮 Servo Motor Connections  
| Servo Direction | Arduino Pin | Jetson Nano Pin |
|------------------|--------------|------------------|
| Left             | 8            | 24               |
| Right            | 12           | 32               |
| Up               | 13           | 33               |
| Down             | 7            | 26               |

### 💡 Light (LED) Connections  
| LED Direction | Arduino Pin | Jetson Nano Pin |
|----------------|--------------|------------------|
| Left           | 11           | 23               |
| Right          | 10           | 19               |
| Up             | 9            | 21               |
| Down           | 6            | 31               |

---

## 👁️ Main Eye Tracker Instructions

- In your Python eye tracker script, **update the video stream URL on line 93** with the IP address you copied from the ESP32 serial output.
- Ensure the **ESP32-CAM has a clear view of the user's entire face** for optimal tracking.
