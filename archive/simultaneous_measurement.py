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
import pickle
# %%

status = {}
chandle = ctypes.c_int16()

# Opens the device/s
status["openunit"] = ps.ps3000aOpenUnit(ctypes.byref(chandle), None)

try:
    assert_pico_ok(status["openunit"])
except:

    # powerstate becomes the status number of openunit
    powerstate = status["openunit"]

    # If powerstate is the same as 282 then it will run this if statement
    if powerstate == 282:
        # Changes the power input to "PICO_POWER_SUPPLY_NOT_CONNECTED"
        status["ChangePowerSource"] = ps.ps3000aChangePowerSource(chandle, 282)
        # If the powerstate is the same as 286 then it will run this if statement
    elif powerstate == 286:
        # Changes the power input to "PICO_USB3_0_DEVICE_NON_USB3_0_PORT"
        status["ChangePowerSource"] = ps.ps3000aChangePowerSource(chandle, 286)
    else:
        raise

    assert_pico_ok(status["ChangePowerSource"])

# Set up channel A
# handle = chandle
# channel = PS3000A_CHANNEL_A = 0
# enabled = 1
# coupling type = PS3000A_DC = 1
# range = PS3000A_5V = 8
# analogue offset = 0 V
chARange = 6
status["setChA"] = ps.ps3000aSetChannel(chandle, 0, 1, 1, chARange, 0)
assert_pico_ok(status["setChA"])

# Finds the max ADC count
# Handle = chandle
# Value = ctypes.byref(maxADC)
maxADC = ctypes.c_int16()
status["maximumValue"] = ps.ps3000aMaximumValue(chandle, ctypes.byref(maxADC))
assert_pico_ok(status["maximumValue"])

# Set an advanced trigger
adcTriggerLevel = mV2adc(500, chARange, maxADC)

# set trigger channel properties
# handle = chandle
channelProperties = ps.PS3000A_TRIGGER_CHANNEL_PROPERTIES(adcTriggerLevel,
                                                          10,
                                                          adcTriggerLevel,
                                                          10,
                                                          ps.PS3000A_CHANNEL["PS3000A_CHANNEL_A"],
                                                          ps.PS3000A_THRESHOLD_MODE["PS3000A_LEVEL"])
nChannelProperties = 1
# auxOutputEnabled = 0
autoTriggerMilliseconds = 10000
status["setTrigProp"] = ps.ps3000aSetTriggerChannelProperties(chandle, ctypes.byref(channelProperties), nChannelProperties, 0, autoTriggerMilliseconds)
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
status["setTrigCond"] = ps.ps3000aSetTriggerChannelConditionsV2(chandle, ctypes.byref(conditions), nConditions)
assert_pico_ok(status["setTrigCond"])

# set trigger directions
# handle = chandle
channelADirection = ps.PS3000A_THRESHOLD_DIRECTION["PS3000A_RISING"]
channelBDirection = ps.PS3000A_THRESHOLD_DIRECTION["PS3000A_NONE"]
channelCDirection = ps.PS3000A_THRESHOLD_DIRECTION["PS3000A_NONE"]
channelDDirection = ps.PS3000A_THRESHOLD_DIRECTION["PS3000A_NONE"]
extDirection = ps.PS3000A_THRESHOLD_DIRECTION["PS3000A_RISING"]
auxDirection = ps.PS3000A_THRESHOLD_DIRECTION["PS3000A_NONE"]
status["setTrigDir"] = ps.ps3000aSetTriggerChannelDirections(chandle, channelADirection, channelBDirection, channelCDirection, channelDDirection, extDirection, auxDirection)
assert_pico_ok(status["setTrigDir"])

# Setting the number of sample to be collected
preTriggerSamples = 1000
postTriggerSamples = 30000
maxsamples = preTriggerSamples + postTriggerSamples


# Output a square wave with peak-to-peak voltage of 2 V and frequency of 10 kHz
# handle = chandle
# offsetVoltage = -1000000
# pkToPk = 1500000
# waveType = ctypes.c_int16(1) = PS3000A_SQUARE
# startFrequency = 10 kHz
# stopFrequency = 10 kHz
# increment = 0
# dwellTime = 1
# sweepType = ctypes.c_int16(1) = PS3000A_UP
# operation = 0
# shots = 0
# sweeps = 0
# triggerType = ctypes.c_int16(0) = PS3000A_SIGGEN_RISING
# triggerSource = ctypes.c_int16(0) = P3000A_SIGGEN_NONE

# triggerSource = ctypes.c_int16(0) = P3000A_SIGGEN_NONE

# extInThreshold = 1
wavetype = ctypes.c_int16(1)
sweepType = ctypes.c_int32(0)
triggertype = ctypes.c_int32(0)
triggerSource = ctypes.c_int32(4)

status["SetSigGenBuiltIn"] = ps.ps3000aSetSigGenBuiltIn(chandle, -000000, 2000000, wavetype, 1000, 1000, 0, 0, sweepType, 0, 1, 0, triggertype, triggerSource, 0)
assert_pico_ok(status["SetSigGenBuiltIn"])



# Gets timebase innfomation
# WARNING: When using this example it may not be possible to access all Timebases as all channels are enabled by default when opening the scope.  
# To access these Timebases, set any unused analogue channels to off.
# Handle = chandle
# Timebase = 2 = timebase
# Nosample = maxsamples
# TimeIntervalNanoseconds = ctypes.byref(timeIntervalns)
# MaxSamples = ctypes.byref(returnedMaxSamples)
# Segement index = 0
timebase = 2
timeIntervalns = ctypes.c_float()
returnedMaxSamples = ctypes.c_int16()
status["GetTimebase"] = ps.ps3000aGetTimebase2(chandle, timebase, maxsamples, ctypes.byref(timeIntervalns), 1, ctypes.byref(returnedMaxSamples), 0)
assert_pico_ok(status["GetTimebase"])

# Creates a overlow location for data
overflow = ctypes.c_int16()
# Creates converted types maxsamples
cmaxSamples = ctypes.c_int32(maxsamples)



i2c = busio.I2C(board.SCL, board.SDA)
sensor = adafruit_bno055.BNO055_I2C(i2c)
pitch = []
roll = []
previous_angle = [0,0,0]
all_angles = []
length = 1000
correction = 180
all_times = []


all_samples = []
start = time.time()
num_averaged_samples = 1000
for idx in range(num_averaged_samples):

    angle = sensor.euler
    current_angle = list(angle)
    if None in current_angle:
        continue

    status["runblock"] = ps.ps3000aRunBlock(chandle, preTriggerSamples, postTriggerSamples, timebase, 1, None, 0, None, None)
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
    status["SetDataBuffers"] = ps.ps3000aSetDataBuffers(chandle, 0, ctypes.byref(bufferAMax), ctypes.byref(bufferAMin), maxsamples, 0, 0)
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

    status["GetValues"] = ps.ps3000aGetValues(chandle, 0, ctypes.byref(cmaxSamples), 0, 0, 0, ctypes.byref(overflow))
    assert_pico_ok(status["GetValues"])

    # Converts ADC from channel A to mV
    adc2mVChAMax =  adc2mV(bufferAMax, chARange, maxADC)
    all_samples.append(adc2mVChAMax)
    
    


    if current_angle[2] < 0:
        correction *= -1

    current_angle[2] = current_angle[2]-correction

    # if abs(current_angle[0]) > 90:
    #     current_angle[0] = previous_angle[0]
    # if abs(current_angle[1]) > 90:
    #     current_angle[1] = previous_angle[1]
    # if abs(current_angle[2]) > 90:
    #     current_angle[2] = previous_angle[2]

    # setting data is about < 1ms
    pitch.append(current_angle[1])
    roll.append(current_angle[2])
    print(f'pitch: {current_angle[1]}, roll: {current_angle[2]}')

all_recordings = [all_samples,pitch,roll]

now = datetime.now()
formatted_date_time = now.strftime("%Y-%m-%d %H:%M:%S")
file_name = f"simultaneous_{formatted_date_time}.pkl"
with open(file_name,'wb') as file:
    pickle.dump(all_recordings,file)

averaged_samples = np.mean(all_samples,axis=0)
end = time.time()
print(f'processing {num_averaged_samples}: {end-start:0.4f}s')

# Creates the time data
time = np.linspace(0, (cmaxSamples.value - 1) * timeIntervalns.value, cmaxSamples.value)
distance = np.arange(len(averaged_samples))/(4*1000)*13/10
plt.plot(distance,averaged_samples)
plt.show()

status["stop"] = ps.ps3000aStop(chandle)
assert_pico_ok(status["stop"])

# Closes the unit
# Handle = chandle
status["close"] = ps.ps3000aCloseUnit(chandle)
assert_pico_ok(status["close"])



# %%

with open('/home/soundpass/Code/simultaneous_2025-02-08 16:33:50.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

# %%
