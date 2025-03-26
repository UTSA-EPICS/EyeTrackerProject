import pyglet
import time

sound = pyglet.media.load("../Sounds/blink_FNZ3zVv.mp3", streaming = False)
left_sound = pyglet.media.load("../Sounds/timer_beep.mp3", streaming = False)
right_sound = pyglet.media.load("../Sounds/short-beep.mp3", streaming = False)


right_sound.play()
time.sleep(1)

