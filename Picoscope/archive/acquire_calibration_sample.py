# %%
# Copyright (C) 2018 Pico Technology Ltd. See LICENSE file for terms.
#
# PS3000A BLOCK MODE EXAMPLE
# This example opens a 3000a driver device, sets up one channels and a trigger then collects a block of data.
# This data is then plotted as mV against time in ns.

import ctypes
from picosdk.ps3000a import ps3000a as ps
import numpy as np
import matplotlib.pyplot as plt
from picosdk.functions import adc2mV, mV2adc, assert_pico_ok
import time
from datetime import datetime
from glob import glob
from scipy import signal 
from scipy.fft import fft, fftfreq
from numpy.lib.stride_tricks import sliding_window_view

# Create chandle and status ready for use
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
postTriggerSamples = 39000
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


# Starts the block capture
# Handle = chandle
# Number of prTriggerSamples
# Number of postTriggerSamples
# Timebase = 2 = 4ns (see Programmer's guide for more information on timebases)
# time indisposed ms = None (This is not needed within the example)
# Segment index = 0
# LpRead = None
# pParameter = None
all_samples = []
start = time.time()
num_averaged_samples = 50
for idx in range(num_averaged_samples):
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

averaged_samples = np.mean(all_samples,axis=0)
end = time.time()
print(f'processing {num_averaged_samples}: {end-start}s')

# Creates the time data
time = np.linspace(0, (cmaxSamples.value - 1) * timeIntervalns.value, cmaxSamples.value)

# Plots the data from channel A onto a graph
# for idx,sample in enumerate(all_samples):
#     plt.plot(time, sample)

# plt.xlabel('Time (ns)')
# plt.ylabel('Voltage (mV)')
# plt.show()

# plt.plot(averaged_samples)
# plt.xlabel('Time (ns)')
# plt.ylabel('Voltage (mV)')
# plt.show()

# plt.plot(averaged_samples[10000:100000])
# plt.xlabel('Time (ns)')
# plt.ylabel('Voltage (mV)')
# plt.show()
# %%

sos = signal.butter(10, 20, 'lp', fs=2025, output='sos')
window_size = 1000
filtered_samples = signal.sosfilt(sos, averaged_samples)
windowed_samples = np.zeros(len(filtered_samples))
windowed_std = np.zeros(len(filtered_samples))
windowed_filtered = sliding_window_view(filtered_samples, window_shape=window_size)
moving_average = np.mean(np.abs(windowed_filtered),axis=1)
moving_std = np.std(np.abs(windowed_filtered),axis=1)
half_window_size = int(window_size/2)
windowed_samples[half_window_size:-half_window_size] = moving_average[:-1]
windowed_std[half_window_size:-half_window_size] = moving_std[:-1]
threshold_values = (windowed_samples + windowed_std)*1.2
thresholded_samples = filtered_samples > threshold_values
base_threshold = filtered_samples > 1
combined_threshold = (thresholded_samples * base_threshold)*10
distance = np.arange(len(averaged_samples))/(4*1000)*13/10

plt.plot(distance,averaged_samples,alpha=0.5)
plt.plot(distance,filtered_samples,alpha=0.5)
# plt.plot(distance,windowed_samples,alpha=0.5)
# plt.plot(distance,threshold_values,alpha=0.5)
plt.legend(['averaged','filtered','windowed','threshold'])
# plt.plot(combined_threshold)
plt.xlabel('Time (ns)')
plt.ylabel('Voltage (mV)')
plt.ylim(-70,70)
plt.show()

# plt.plot(averaged_samples[10000:100000])
# plt.plot(filtered_samples[10000:100000])
# plt.plot(windowed_samples[10000:100000])
# plt.plot(threshold_values[10000:100000])
# plt.plot(combined_threshold[10000:100000])

# plt.xlabel('Time (ns)')
# plt.ylabel('Voltage (mV)')
# plt.show()

status["stop"] = ps.ps3000aStop(chandle)
assert_pico_ok(status["stop"])

# Closes the unit
# Handle = chandle
status["close"] = ps.ps3000aCloseUnit(chandle)
assert_pico_ok(status["close"])
now = datetime.now()

# %%

# fourier = fft(averaged_samples)
# freq = fftfreq(len(averaged_samples), d=4e-9)
# freq_len = int(len(fourier)/2)
# plt.plot(freq[:freq_len],np.abs(fourier)[:freq_len])
# plt.ylim(-1,10000)

# Stops the scope
# Handle = chandle
status["stop"] = ps.ps3000aStop(chandle)
assert_pico_ok(status["stop"])

# Closes the unit
# Handle = chandle
status["close"] = ps.ps3000aCloseUnit(chandle)
assert_pico_ok(status["close"])
now = datetime.now()

formatted_date_time = now.strftime("%Y-%m-%d %H:%M:%S")
file_name = f"Tofu_calibration_{formatted_date_time}.csv"
# np.savetxt(file_name, averaged_samples, delimiter=",")

# print(f'saved calibration sample as {file_name}')
# Displays the staus returns
print(status)

# %%
cal_names = glob('/home/soundpass/Code/Tofu*.csv')

for cal_name in cal_names:
    calibration_array = np.genfromtxt(cal_name)
    plt.plot(calibration_array)
plt.show()

for cal_name in cal_names:
    calibration_array = np.genfromtxt(cal_name)
    plt.plot(calibration_array[10000:100000])
plt.show()

# %%
