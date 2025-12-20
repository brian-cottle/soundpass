import time
import board
import busio
import adafruit_bno055
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

# %matplotlib inline
import numpy as np
import math

i2c = busio.I2C(board.SCL, board.SDA)
sensor = adafruit_bno055.BNO055_I2C(i2c)


previous_angle = [0,0,0]
all_angles = []
length = 1000

########################################

# Create figure for plotting
fig = plt.figure(figsize=(10,7))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
plt.show(block=False)

pitch, = ax1.plot([], 'r-')
roll, = ax2.plot([], 'b-')

ax1.set_title('pitch')
ax1.set_xlim((-2000,2000))
ax1.set_ylim((-1000,0))
ax1.set_xticklabels('')
ax1.set_yticklabels('')


ax2.set_title('roll')
ax2.set_xlim((-2000,2000))
ax2.set_ylim((-1000,0))
text = ax2.text(0,-500, '')
ax2.set_xticklabels('')
ax2.set_yticklabels('')

t_start = time.time()

def get_line_x(angle):
    end_x = 1000*math.tan(np.radians(abs(angle)))

    end_x *= math.copysign(1,angle)

    return(end_x)


def animate(i,ax1,ax2,pitch,roll,text,t_start,previous_angle):
    correction = 180
    ## angle stuff is about 15ms
    current_angle = list(sensor.euler)
    if None in current_angle:
        return()

    if current_angle[2] < 0:
        correction *= -1

    current_angle[2] = current_angle[2]-correction

    if abs(current_angle[0]) > 90:
        # current_angle[0] = current_angle[0] - 360
        # if current_angle[0] > 180:
        current_angle[0] = previous_angle[0]
    if abs(current_angle[1]) > 90:
        current_angle[1] = previous_angle[1]
    if abs(current_angle[2]) > 90:
        current_angle[2] = previous_angle[2]

    # setting data is about < 1ms
    pitch.set_data([0,get_line_x(current_angle[1])],[0,-1000])
    roll.set_data([0,get_line_x(current_angle[2])],[0,-1000])
    


    tx = 'Mean Frame Rate:\n {fps:.3f}FPS'.format(fps= ((i+1) / (time.time() - t_start)) )
    text.set_text(tx)

    # ax1.draw_artist(pitch)
    # ax2.draw_artist(roll)
    # ax2.draw_artist(text)

    fig.canvas.blit(ax1.bbox)
    fig.canvas.blit(ax2.bbox)

    fig.canvas.flush_events()


    previous_angle = current_angle

# Set up plot to call animate() function periodically
ani = FuncAnimation(fig, animate, fargs=(ax1,ax2,pitch,roll,text,t_start,previous_angle), interval=10)
plt.show()
