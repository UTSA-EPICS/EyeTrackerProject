import Jetson.GPIO as GPIO
import time


# Pin Definitions
led_pin = 18  # Use the pin number for the GPIO (BCM numbering)


# Set up GPIO
GPIO.setmode(GPIO.BCM)  # Use BCM pin numbering
GPIO.setup(led_pin, GPIO.OUT, initial=GPIO.LOW)  # Set pin as output and initialize it to LOW


#   Servo GPIO pin setup
servo_pin = 18  # GPIO pin connected to the servo (BCM numbering)
min_duty_cycle = 2.5  # Min duty cycle for 0 degrees
max_duty_cycle = 12.5  # Max duty cycle for 180 degrees


def set_servo_angle(angle):
   """Set the servo to a specific angle using software PWM."""
   duty_cycle = min_duty_cycle + (max_duty_cycle - min_duty_cycle) * (angle / 180.0)
