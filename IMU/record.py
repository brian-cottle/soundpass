# %%
import time
import board
import busio
import adafruit_bno055
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

# %matplotlib inline
import numpy as np
import math

def get_line_x(angle):
    end_x = 1000*math.tan(np.radians(abs(angle)))

    end_x *= math.copysign(1,angle)

    return(end_x)

t_start = time.time()

i2c = busio.I2C(board.SCL, board.SDA)
sensor = adafruit_bno055.BNO055_I2C(i2c)

pitch = []
roll = []
previous_angle = [0,0,0]
all_angles = []
length = 1000
correction = 180
all_times = []


while time.time() - t_start < 60:

    ## angle stuff is about 15ms
    angle = sensor.euler
    print(angle)
    current_angle = list(angle)
    if None in current_angle:
        continue

    if current_angle[2] < 0:
        correction *= -1

    current_angle[2] = current_angle[2]-correction

    if abs(current_angle[0]) > 90:
        current_angle[0] = previous_angle[0]
    if abs(current_angle[1]) > 90:
        current_angle[1] = previous_angle[1]
    if abs(current_angle[2]) > 90:
        current_angle[2] = previous_angle[2]

    # setting data is about < 1ms
    pitch.append(current_angle[1])
    roll.append(current_angle[2])
    all_times.append(time.time())
    print(f'{pitch[-1]},{roll[-1]}')
    time.sleep(0.01)
    previous_angle = current_angle
# %%
# save_data = np.asarray([pitch,roll,all_times])
# print(save_data)
# np.savetxt('orientation.out', save_data, delimiter=',') 