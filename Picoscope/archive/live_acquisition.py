# %%
from pynput import keyboard
# %%
import ctypes
from picosdk.ps3000a import ps3000a as ps
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from picosdk.functions import adc2mV, mV2adc, assert_pico_ok
import time
import board
import busio
import adafruit_bno055
import numpy as np
import math
from glob import glob
from datetime import datetime
from matplotlib.gridspec import GridSpec
from scipy import signal 
from scipy import signal 
from scipy.fft import fft, fftfreq
from numpy.lib.stride_tricks import sliding_window_view
import pickle

preTriggerSamples = 1000 # Hz
postTriggerSamples = 39000 # Hz
timebase = 2 # 2ns per sample
nChannelProperties = 1
autoTriggerMilliseconds = 10000 # 10s before auto triggering
chARange = 6 # 1V
advanced_trigger_threshold = 500 # mV
sig_gen_peak_to_peak = 2000000 # 2V
sig_gen_hz = 1000
sig_gen_offset = 00000000 # 0V
num_samples = 5 # num samples to average for live feed
save_array = False
pitch_angles = [0]
roll_angles = [0]
yaw_angles = [0]
angle_set = False
set_pitch = 0
set_roll = 0
set_yaw = 0
saved_pitch = 0
saved_roll = 0
saved_yaw = 0
single_save = False
continuous_save = False
change_continuous_save = False
switch_angle = False
save_text = ''
annotation = ''

now = datetime.now()
formatted_date_time = now.strftime("%Y-%m-%d %H:%M:%S")
save_name = f"tofu_balloon_{formatted_date_time}.csv"

# %% ------------------------------------------------------------------------- #

def on_key_press(event):
    global angle_set
    global continuous_save
    global single_save
    global save_text
    global annotation

    if event.key == 'a':
            print(f'setting angle at:')
            print(f'pitch: {set_pitch}, roll: {set_roll}, yaw: {set_yaw}')

            angle_set = True

    if event.key == 'b':
            print(f'opening angle acquisition')
            angle_set = False

    if event.key == 'p':
        print('saving single spectrum')
        save_text = input('input file name, press enter when done: ')
        annotation = input('input annotation, press enter when done: ')
        single_save = True
        # single save

    if event.key == 'c':
        # continuous save
        if not continuous_save:
            save_text = input('input file name, press enter when done: ')
            try:
                annotation = input('input annotation, press enter when done: ')
            except RuntimeError:
                print('Unknown error, did not save annotation')
        continuous_save = not continuous_save
        print(f'continuous save: {continuous_save}')

    elif event.key == 'd':
        status["stop"] = ps.ps3000aStop(chandle)
        assert_pico_ok(status["stop"])

        # Closes the unit
        # Handle = chandle
        status["close"] = ps.ps3000aCloseUnit(chandle)
        assert_pico_ok(status["close"])

        ani.event_source.stop()


# ---------------------------------------------------------------------------- #

def get_line_x(angle):
    end_x = 1000*math.tan(np.radians(abs(angle)))

    end_x *= math.copysign(1,angle)

    return(end_x)


# ---------------------------------------------------------------------------- #

def acquire_multiple_signals(num_samples,
                             sensor,
                             chandle,
                             chARange, 
                             preTriggerSamples, 
                             postTriggerSamples, 
                             timebase):
    all_samples = []

    for idx in range(num_samples):
        angle = sensor.euler
        current_angle = list(angle)
        if None in current_angle:
            continue

        status["runblock"] = ps.ps3000aRunBlock(chandle, 
                                                preTriggerSamples, 
                                                postTriggerSamples, 
                                                timebase, 
                                                1, 
                                                None, 
                                                0, 
                                                None, 
                                                None)
        assert_pico_ok(status["runblock"])
        status["sigGenSoftware"] = ps.ps3000aSigGenSoftwareControl(chandle,0)

        # Create buffers ready for assigning pointers for data collection
        bufferAMax = (ctypes.c_int16 * maxsamples)()
        bufferAMin = (ctypes.c_int16 * maxsamples)() # used for downsampling which isn't in the scope of this example

        # Setting the data buffer location for data collection from channel A
        # Handle = Chandle
        # source = ps3000A_channel_A = 0
        # Buffer max = ctypes.byref(bufferAMax)
        # Buffer min = ctypes.byref(bufferAMin)
        # Buffer length = maxsamples
        # Segment index = 0
        # Ratio mode = ps3000A_Ratio_Mode_None = 0
        status["SetDataBuffers"] = ps.ps3000aSetDataBuffers(chandle, 
                                                            0, 
                                                            ctypes.byref(bufferAMax), 
                                                            ctypes.byref(bufferAMin), 
                                                            maxsamples, 
                                                            0, 
                                                            0)

        assert_pico_ok(status["SetDataBuffers"])

        # Creates a overlow location for data
        overflow = (ctypes.c_int16 * 10)()
        # Creates converted types maxsamples
        cmaxSamples = ctypes.c_int32(maxsamples)

        # Checks data collection to finish the capture
        ready = ctypes.c_int16(0)
        check = ctypes.c_int16(0)
        while ready.value == check.value:
            status["isReady"] = ps.ps3000aIsReady(chandle, ctypes.byref(ready))

        # Handle = chandle
        # start index = 0
        # noOfSamples = ctypes.byref(cmaxSamples)
        # DownSampleRatio = 0
        # DownSampleRatioMode = 0
        # SegmentIndex = 0
        # Overflow = ctypes.byref(overflow)

        status["GetValues"] = ps.ps3000aGetValues(chandle, 
                                                  0, 
                                                  ctypes.byref(cmaxSamples), 
                                                  0, 
                                                  0, 
                                                  0, 
                                                  ctypes.byref(overflow))

        assert_pico_ok(status["GetValues"])

        # Converts ADC from channel A to mV
        adc2mVChAMax =  adc2mV(bufferAMax, chARange, maxADC)
        all_samples.append(adc2mVChAMax)

        return(all_samples)


# ---------------------------------------------------------------------------- #

def setup_IMU():
    i2c = busio.I2C(board.SCL, board.SDA)
    sensor = adafruit_bno055.BNO055_I2C(i2c)

    return(sensor)


# ---------------------------------------------------------------------------- #

def animate(i,
            sensor,
            ax1,
            ax2,
            ax3,
            ax4,
            pitch,
            pitch_mean,
            sample,
            threshold,
            segmentation,
            roll,
            roll_mean,
            yaw,
            yaw_mean,
            text,
            t_start,
            previous_angle,
            chandle,
            chARange, 
            timebase,
            distance,
            # cal_array,
            preTriggerSamples=1000, 
            postTriggerSamples=39000, 
            num_samples=5,):
    global pitch_angles
    global roll_angles
    global yaw_angles
    global angle_set
    global set_pitch
    global set_roll
    global set_yaw
    global saved_pitch
    global saved_roll
    global saved_yaw
    global single_save
    global continuous_save
    global switch_angle
    global save_text
    global annotation


    sos = signal.butter(10, 20, 'lp', fs=2025, output='sos')  
    correction = 180
    ## angle stuff is about 15ms
    current_angle = list(sensor.euler)
    if None in current_angle:
        return()

    if current_angle[2] < 0:
        correction *= -1

    current_angle[2] = current_angle[2]-correction

    # if abs(current_angle[0]) > 90:
    #     # current_angle[0] = current_angle[0] - 360
    #     # if current_angle[0] > 180:
    #     current_angle[0] = previous_angle[0]

    if abs(current_angle[1]) > 90:
        current_angle[1] = previous_angle[1]
    if abs(current_angle[2]) > 90:
        current_angle[2] = previous_angle[2]

    current_angle[0] = np.radians(current_angle[0])
    pitch_angles.append(current_angle[1]*1.5)
    roll_angles.append(current_angle[2]*1.5)
    yaw_angles.append(current_angle[0])

    if len(pitch_angles) > 10:
        pitch_angles.pop(0)

    if len(roll_angles) > 10:
        roll_angles.pop(0)

    if len(yaw_angles) > 10:
        yaw_angles.pop(0)
        
    if not angle_set:

        set_pitch = np.mean(pitch_angles)
        set_roll = np.mean(roll_angles)
        set_yaw = np.mean(yaw_angles)

        pitch_mean.set_data([0,get_line_x(set_pitch)],[0,-1000])
        roll_mean.set_data([0,get_line_x(set_roll)],[0,-1000])
        yaw_mean.set_data([set_yaw,set_yaw],[1,0])
        switch_angle = False
        

    if angle_set and not switch_angle:
        set_pitch = np.mean(pitch_angles)
        set_roll = np.mean(roll_angles)
        set_yaw = np.mean(yaw_angles)
		
        saved_pitch = set_pitch
        saved_roll = set_roll
        saved_yaw = set_yaw
		
    if angle_set:
        pitch_mean.set_data([0,get_line_x(saved_pitch)],[0,-1000])
        roll_mean.set_data([0,get_line_x(saved_roll)],[0,-1000])
        yaw_mean.set_data([saved_yaw,saved_yaw],[1,0])
        switch_angle = True

    # setting data is about < 1ms
    pitch.set_data([0,get_line_x(pitch_angles[-1])],[0,-1000])
    roll.set_data([0,get_line_x(roll_angles[-1])],[0,-1000])
    yaw.set_data([yaw_angles[-1],yaw_angles[-1]],[1,0])
    

    output_samples = []

    for idx in range(num_samples):
        angle = sensor.euler
        current_angle = list(angle)
        if None in current_angle:
            continue

        status["runblock"] = ps.ps3000aRunBlock(chandle, 
                                                preTriggerSamples, 
                                                postTriggerSamples, 
                                                timebase, 
                                                1, 
                                                None, 
                                                0, 
                                                None, 
                                                None)
        assert_pico_ok(status["runblock"])
        status["sigGenSoftware"] = ps.ps3000aSigGenSoftwareControl(chandle,0)

        # Create buffers ready for assigning pointers for data collection
        bufferAMax = (ctypes.c_int16 * maxsamples)()
        bufferAMin = (ctypes.c_int16 * maxsamples)() # used for downsampling which isn't in the scope of this example

        # Setting the data buffer location for data collection from channel A
        # Handle = Chandle
        # source = ps3000A_channel_A = 0
        # Buffer max = ctypes.byref(bufferAMax)
        # Buffer min = ctypes.byref(bufferAMin)
        # Buffer length = maxsamples
        # Segment index = 0
        # Ratio mode = ps3000A_Ratio_Mode_None = 0
        status["SetDataBuffers"] = ps.ps3000aSetDataBuffers(chandle, 
                                                            0, 
                                                            ctypes.byref(bufferAMax), 
                                                            ctypes.byref(bufferAMin), 
                                                            maxsamples, 
                                                            0, 
                                                            0)

        assert_pico_ok(status["SetDataBuffers"])

        # Creates a overlow location for data
        overflow = (ctypes.c_int16 * 10)()
        # Creates converted types maxsamples
        cmaxSamples = ctypes.c_int32(maxsamples)

        # Checks data collection to finish the capture
        ready = ctypes.c_int16(0)
        check = ctypes.c_int16(0)
        while ready.value == check.value:
            status["isReady"] = ps.ps3000aIsReady(chandle, ctypes.byref(ready))

        # Handle = chandle
        # start index = 0
        # noOfSamples = ctypes.byref(cmaxSamples)
        # DownSampleRatio = 0
        # DownSampleRatioMode = 0
        # SegmentIndex = 0
        # Overflow = ctypes.byref(overflow)

        status["GetValues"] = ps.ps3000aGetValues(chandle, 
                                                  0, 
                                                  ctypes.byref(cmaxSamples), 
                                                  0, 
                                                  0, 
                                                  0, 
                                                  ctypes.byref(overflow))

        assert_pico_ok(status["GetValues"])

        # Converts ADC from channel A to mV
        adc2mVChAMax =  adc2mV(bufferAMax, chARange, maxADC)
        output_samples.append(adc2mVChAMax)

    # output_samples = acquire_multiple_signals(num_samples,
    #                             sensor,
    #                             chandle,
    #                             chARange, 
    #                             preTriggerSamples, 
    #                             postTriggerSamples, 
    #                             timebase)

    # if save_array:
    #     np.savetxt(save_name, np.asarray(all_samples), delimiter=",")
    #     print(f'saved sample: {save_name}')
    #     save_array = False
    try:
        averaged_samples = np.mean(output_samples,axis=0)
        filtered_samples = signal.sosfilt(sos, averaged_samples)
    except ValueError:
        return()

    if single_save:
        now = datetime.now()
        formatted_date_time = int(time.time()*1000)
        file_name = f"{save_text}_{formatted_date_time}.pkl"
        with open(file_name,'wb') as file:
            pickle.dump([averaged_samples,current_angle,annotation],file)
        print(f'saved: {file_name}')

        single_save = False


    if continuous_save:
        now = datetime.now()
        formatted_date_time = int(time.time()*1000)
        file_name = f"{save_text}_{formatted_date_time}.pkl"
        with open(file_name,'wb') as file:
            pickle.dump([averaged_samples,current_angle,annotation],file)
        print(f'saved: {file_name}')


    # sos = signal.butter(10, 10, 'hp', fs=2025, output='sos')
    # window_size = 1000
    # filtered = signal.sosfilt(sos, averaged_samples)
    # windowed_samples = np.zeros(len(filtered))
    # windowed_std = np.zeros(len(filtered))
    # windowed_filtered = sliding_window_view(filtered, window_shape=window_size)
    # moving_average = np.mean(np.abs(windowed_filtered),axis=1)
    # moving_std = np.std(np.abs(windowed_filtered),axis=1)
    # half_window_size = int(window_size/2)
    # windowed_samples[half_window_size:-half_window_size] = moving_average[:-1]
    # windowed_std[half_window_size:-half_window_size] = moving_std[:-1]
    # threshold_values = (windowed_samples + windowed_std)*1.2
    # thresholded_samples = filtered_samples > threshold_values
    # base_threshold = filtered_samples > 1
    # combined_threshold = (thresholded_samples * base_threshold)*10
    distance = np.arange(len(averaged_samples))/(4*1000)*13/10


    sample.set_data(distance,filtered_samples)
    # threshold.set_data(distance,threshold_values)
    # segmentation.set_data(distance,combined_threshold)

    tx = 'Mean Frame Rate:\n {fps:.3f}FPS'.format(fps= ((i+1) / (time.time() - t_start)) )
    text.set_text(tx)

    # ax1.draw_artist(pitch)
    # ax2.draw_artist(roll)
    # ax2.draw_artist(text)

    fig.canvas.blit(ax1.bbox)
    fig.canvas.blit(ax2.bbox)
    fig.canvas.blit(ax3.bbox)

    fig.canvas.flush_events()


    previous_angle = current_angle

# ---------------------------------------------------------------------------- #




status = {}
chandle = ctypes.c_int16()

# Opens the device/s
status["openunit"] = ps.ps3000aOpenUnit(ctypes.byref(chandle), None)

try:
    assert_pico_ok(status["openunit"])
except:
    powerstate = status["openunit"]

    if powerstate == 282:
        status["ChangePowerSource"] = ps.ps3000aChangePowerSource(chandle, 282)
    elif powerstate == 286:
        status["ChangePowerSource"] = ps.ps3000aChangePowerSource(chandle, 286)
    else:
        raise

    assert_pico_ok(status["ChangePowerSource"])


status["setChA"] = ps.ps3000aSetChannel(chandle, 0, 1, 1, chARange, 0)
assert_pico_ok(status["setChA"])


maxADC = ctypes.c_int16()
status["maximumValue"] = ps.ps3000aMaximumValue(chandle, ctypes.byref(maxADC))
assert_pico_ok(status["maximumValue"])

# Set an advanced trigger
adcTriggerLevel = mV2adc(advanced_trigger_threshold, chARange, maxADC)

channelProperties = ps.PS3000A_TRIGGER_CHANNEL_PROPERTIES(adcTriggerLevel,
                                                          10,
                                                          adcTriggerLevel,
                                                          10,
                                                          ps.PS3000A_CHANNEL["PS3000A_CHANNEL_A"],
                                                          ps.PS3000A_THRESHOLD_MODE["PS3000A_LEVEL"])

status["setTrigProp"] = ps.ps3000aSetTriggerChannelProperties(chandle, 
                                                              ctypes.byref(channelProperties), 
                                                              nChannelProperties, 
                                                              0, 
                                                              autoTriggerMilliseconds)
assert_pico_ok(status["setTrigProp"])

# set trigger conditions V2
# chandle = handle
conditions = ps.PS3000A_TRIGGER_CONDITIONS_V2(ps.PS3000A_TRIGGER_STATE["PS3000A_CONDITION_TRUE"],
                                              ps.PS3000A_TRIGGER_STATE["PS3000A_CONDITION_DONT_CARE"],
                                              ps.PS3000A_TRIGGER_STATE["PS3000A_CONDITION_DONT_CARE"],
                                              ps.PS3000A_TRIGGER_STATE["PS3000A_CONDITION_DONT_CARE"],
                                              ps.PS3000A_TRIGGER_STATE["PS3000A_CONDITION_DONT_CARE"],
                                              ps.PS3000A_TRIGGER_STATE["PS3000A_CONDITION_DONT_CARE"],
                                              ps.PS3000A_TRIGGER_STATE["PS3000A_CONDITION_DONT_CARE"],
                                              ps.PS3000A_TRIGGER_STATE["PS3000A_CONDITION_DONT_CARE"])
nConditions = 1
status["setTrigCond"] = ps.ps3000aSetTriggerChannelConditionsV2(chandle, 
                                                                ctypes.byref(conditions), 
                                                                nConditions)
assert_pico_ok(status["setTrigCond"])

channelADirection = ps.PS3000A_THRESHOLD_DIRECTION["PS3000A_RISING"]
channelBDirection = ps.PS3000A_THRESHOLD_DIRECTION["PS3000A_NONE"]
channelCDirection = ps.PS3000A_THRESHOLD_DIRECTION["PS3000A_NONE"]
channelDDirection = ps.PS3000A_THRESHOLD_DIRECTION["PS3000A_NONE"]
extDirection = ps.PS3000A_THRESHOLD_DIRECTION["PS3000A_RISING"]
auxDirection = ps.PS3000A_THRESHOLD_DIRECTION["PS3000A_NONE"]

status["setTrigDir"] = ps.ps3000aSetTriggerChannelDirections(chandle, 
                                                             channelADirection, 
                                                             channelBDirection, 
                                                             channelCDirection, 
                                                             channelDDirection, 
                                                             extDirection, 
                                                             auxDirection)
assert_pico_ok(status["setTrigDir"])

# Setting the number of sample to be collected
maxsamples = preTriggerSamples + postTriggerSamples

wavetype = ctypes.c_int16(1)
sweepType = ctypes.c_int32(0)
triggertype = ctypes.c_int32(0)
triggerSource = ctypes.c_int32(4)

status["SetSigGenBuiltIn"] = ps.ps3000aSetSigGenBuiltIn(chandle, 
                                                        sig_gen_offset, 
                                                        sig_gen_peak_to_peak, 
                                                        wavetype, 
                                                        sig_gen_hz, 
                                                        sig_gen_hz, 
                                                        0, 
                                                        0, 
                                                        sweepType, 
                                                        0, 
                                                        1, 
                                                        0, 
                                                        triggertype, 
                                                        triggerSource, 
                                                        0)

assert_pico_ok(status["SetSigGenBuiltIn"])


timeIntervalns = ctypes.c_float()
returnedMaxSamples = ctypes.c_int16()
status["GetTimebase"] = ps.ps3000aGetTimebase2(chandle, 
                                               timebase, 
                                               maxsamples, 
                                               ctypes.byref(timeIntervalns), 
                                               1, 
                                               ctypes.byref(returnedMaxSamples), 
                                               0)
assert_pico_ok(status["GetTimebase"])

# Creates a overlow location for data
overflow = ctypes.c_int16()
# Creates converted types maxsamples
cmaxSamples = ctypes.c_int32(maxsamples)


# %% ------------------------------------------------------------------------- # 


fig = plt.figure(figsize=(24, 8))
fig.canvas.mpl_connect('key_press_event', on_key_press)

# Create a 3x3 grid
gs = GridSpec(1, 6, figure=fig)

ax1 = fig.add_subplot(gs[0, 2])
ax2 = fig.add_subplot(gs[0, 0:2])
ax3 = fig.add_subplot(gs[0, 3])
ax4 = fig.add_subplot(gs[0, 4:],projection='polar')

plt.show(block=False)

pitch, = ax1.plot([], 'r-')
pitch_mean, = ax1.plot([], 'k-',linewidth=15,alpha=0.3)
sample, = ax2.plot([],'g-',linewidth=0.2)
threshold, = ax2.plot([],'k-')
segmentation, = ax2.plot([],'c-')
roll, = ax3.plot([], 'b-')
roll_mean, = ax3.plot([], 'k-',linewidth=15,alpha=0.3)
yaw, = ax4.plot([],[], 'r-')
yaw_mean, = ax4.plot([],[], 'k-',linewidth=15,alpha=0.3)

ax1.set_title('Rock',fontsize=25)
ax1.set_xlim((-2000,2000))
ax1.set_ylim((-1000,0))
ax1.set_xticklabels('')
ax1.set_yticklabels('')

ax2.set_title('Signal',fontsize=25)
ax2.set_xlim((0,8))
ax2.set_ylim((-500,500))
ax2.tick_params(axis='x',labelsize=15)
ax2.grid(axis='x', linestyle='--', color='gray', alpha=0.7)

ax3.set_title('Sweep',fontsize=25)
ax3.set_xlim((-2000,2000))
ax3.set_ylim((-1000,0))
text = ax2.text(0,-500, '')
ax3.set_xticklabels('')
ax3.set_yticklabels('')

ax4.set_title('Rotation',fontsize=25)
ax4.set_xticklabels('')
ax4.set_yticklabels('')

t_start = time.time()


# %% ------------------------------------------------------------------------- #


time_separation = np.linspace(0, (cmaxSamples.value - 1) * timeIntervalns.value, cmaxSamples.value)
distance = np.arange((preTriggerSamples + postTriggerSamples))/(4*1000)*13/10

sensor = setup_IMU()
previous_angle = [0,0,0]

# cal_names = glob('/home/soundpass/Code/Tofu_calibration*.csv')
# calibration_arrays = []

# for cal_name in cal_names:
#     calibration_arrays.append(np.genfromtxt(cal_name))

# mean_calibration_array = np.mean(calibration_arrays,axis=0)


ani = FuncAnimation(fig, animate, fargs=(
        sensor,
        ax1,
        ax2,
        ax3,
        ax4,
        pitch,
        pitch_mean,
        sample,
        threshold,
        segmentation,
        roll,
        roll_mean,
        yaw,
        yaw_mean,
        text,
        t_start,
        previous_angle,
        chandle,
        chARange, 
        timebase,
        distance,
        # mean_calibration_array,
        preTriggerSamples, 
        postTriggerSamples, 
        1), interval=30)
plt.show()
