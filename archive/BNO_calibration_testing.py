# %%

import adafruit_bno055
import time
import board
import busio
import math
import numpy as np
import os

# %%

i2c = busio.I2C(board.SCL, board.SDA)
sensor = adafruit_bno055.BNO055_I2C(i2c)
print(sensor.axis_remap)
sensor.axis_remap = (0,1,2,0,1,1)
print(sensor.axis_remap)
while True:
    initial_angle = np.asarray(sensor.euler)
    formatted_angle = [f"{angle:8.3f}" for angle in initial_angle]
    print(f"\rInitial angle: {formatted_angle}", end="", flush=True)


# %%