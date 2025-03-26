import Jetson.GPIO as GPIO
import time

servo_pin = 33  # GPIO pin number for PWM (BOARD pin 12)


frequency = 50  # Servo PWM frequency (50Hz)


MIN_DUTY = 2.5  # Min position (0 degrees)
MAX_DUTY = 12.5  # Max position (180 degrees)

def set_servo_angle(pwm, angle):
    """
    Set the servo to a specific angle.
    :param pwm: PWM instance
    :param angle: Desired angle (0-180)
    """
    duty_cycle = MIN_DUTY + (angle / 180.0) * (MAX_DUTY - MIN_DUTY)
    pwm.ChangeDutyCycle(duty_cycle)
    time.sleep(0.5)  # Give the servo time to move
    pwm.ChangeDutyCycle(0)  # Turn off signal to avoid heating

try:
    # GPIO setup
    GPIO.setmode(GPIO.BOARD)  # Use physical pin numbering
    GPIO.setup(servo_pin, GPIO.OUT)

    # Initialize PWM
    pwm = GPIO.PWM(servo_pin, frequency)
    pwm.start(0)  # Start PWM with 0% duty cycle

    print("Moving servo. Press Ctrl+C to stop.")
    while True:
        # Move to 0 degrees
        print("Angle: 0")
        set_servo_angle(pwm, 0)

        # Move to 90 degrees
        print("Angle: 90")
        set_servo_angle(pwm, 90)

        # Move to 180 degrees
        print("Angle: 180")
        set_servo_angle(pwm, 180)

except KeyboardInterrupt:
    print("Stopping program.")

finally:
    # Cleanup
    pwm.stop()
    GPIO.cleanup()
    print("GPIO cleaned up.")