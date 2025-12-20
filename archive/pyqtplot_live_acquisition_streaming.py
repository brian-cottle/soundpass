import sys
import numpy as np
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
import pyqtgraph as pg
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



def get_line_x(angle):
    end_x = 1000*math.tan(np.radians(abs(angle)))

    end_x *= math.copysign(1,angle)

    return(end_x)


def setup_IMU():
    i2c = busio.I2C(board.SCL, board.SDA)
    sensor = adafruit_bno055.BNO055_I2C(i2c)

    return(sensor)



class TimeSeriesPlotter:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.show()
        self.plot = self.plot_widget.plot(pen='y')
        self.x_data = []
        self.y_data = []
        self.pitch_angles = [0]
        self.roll_angles = [0]
        self.yaw_angles = [0]
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(1)  # Update every 100 ms
        self.i = 0

        self.text_item = pg.TextItem(text="Initial Text", anchor=(5, 100), color=(255, 255, 0),html='<span style="font-size: 16pt;">Initial Text</span>',)

        self.preTriggerSamples = 1000 # Hz
        self.postTriggerSamples = 39000 # Hz
        self.timebase = 2 # 2ns per sample
        self.nChannelProperties = 1
        self.autoTriggerMilliseconds = 10000 # 10s before auto triggering
        self.chARange = 6 # 1V
        self.advanced_trigger_threshold = 500 # mV
        self.sig_gen_peak_to_peak = 2000000 # 2V
        self.sig_gen_hz = 1000
        self.sig_gen_offset = 00000000 # 0V
        self.num_samples = 1 # num samples to average for live feed
        save_array = False
        angle_set = False
        set_pitch = 0
        set_roll = 0
        set_yaw = 0
        self.saved_pitch = 0
        self.saved_roll = 0
        self.saved_yaw = 0
        single_save = False
        continuous_save = False
        change_continuous_save = False
        switch_angle = False
        save_text = ''
        annotation = ''
        self.previous_angle = [0,0,0]
        self.last_time = 0
        self.t_start = time.time()
        self.numBuffersToCapture = 1



        self.status = {}
        self.chandle = ctypes.c_int16()

        self.sensor = setup_IMU()

        # Opens the device/s
        self.status["openunit"] = ps.ps3000aOpenUnit(ctypes.byref(self.chandle), None)

        try:
            assert_pico_ok(self.status["openunit"])
        except:
            powerstate = self.status["openunit"]

            if powerstate == 282:
                self.status["ChangePowerSource"] = ps.ps3000aChangePowerSource(self.chandle, 282)
            elif powerstate == 286:
                self.status["ChangePowerSource"] = ps.ps3000aChangePowerSource(self.chandle, 286)
            else:
                raise

            assert_pico_ok(self.status["ChangePowerSource"])


        self.status["setChA"] = ps.ps3000aSetChannel(self.chandle, 0, 1, 1, self.chARange, 0)
        assert_pico_ok(self.status["setChA"])


        self.maxADC = ctypes.c_int16()
        self.status["maximumValue"] = ps.ps3000aMaximumValue(self.chandle, ctypes.byref(self.maxADC))
        assert_pico_ok(self.status["maximumValue"])

        # Set an advanced trigger
        adcTriggerLevel = mV2adc(self.advanced_trigger_threshold, self.chARange, self.maxADC)

        channelProperties = ps.PS3000A_TRIGGER_CHANNEL_PROPERTIES(adcTriggerLevel,
                                                                10,
                                                                adcTriggerLevel,
                                                                10,
                                                                ps.PS3000A_CHANNEL["PS3000A_CHANNEL_A"],
                                                                ps.PS3000A_THRESHOLD_MODE["PS3000A_LEVEL"])

        self.status["setTrigProp"] = ps.ps3000aSetTriggerChannelProperties(self.chandle, 
                                                                    ctypes.byref(channelProperties), 
                                                                    self.nChannelProperties, 
                                                                    0, 
                                                                    self.autoTriggerMilliseconds)
        assert_pico_ok(self.status["setTrigProp"])

        # set trigger conditions V2
        # self.chandle = handle
        conditions = ps.PS3000A_TRIGGER_CONDITIONS_V2(ps.PS3000A_TRIGGER_STATE["PS3000A_CONDITION_TRUE"],
                                                    ps.PS3000A_TRIGGER_STATE["PS3000A_CONDITION_DONT_CARE"],
                                                    ps.PS3000A_TRIGGER_STATE["PS3000A_CONDITION_DONT_CARE"],
                                                    ps.PS3000A_TRIGGER_STATE["PS3000A_CONDITION_DONT_CARE"],
                                                    ps.PS3000A_TRIGGER_STATE["PS3000A_CONDITION_DONT_CARE"],
                                                    ps.PS3000A_TRIGGER_STATE["PS3000A_CONDITION_DONT_CARE"],
                                                    ps.PS3000A_TRIGGER_STATE["PS3000A_CONDITION_DONT_CARE"],
                                                    ps.PS3000A_TRIGGER_STATE["PS3000A_CONDITION_DONT_CARE"])
        nConditions = 1
        self.status["setTrigCond"] = ps.ps3000aSetTriggerChannelConditionsV2(self.chandle, 
                                                                        ctypes.byref(conditions), 
                                                                        nConditions)
        assert_pico_ok(self.status["setTrigCond"])

        channelADirection = ps.PS3000A_THRESHOLD_DIRECTION["PS3000A_RISING"]
        channelBDirection = ps.PS3000A_THRESHOLD_DIRECTION["PS3000A_NONE"]
        channelCDirection = ps.PS3000A_THRESHOLD_DIRECTION["PS3000A_NONE"]
        channelDDirection = ps.PS3000A_THRESHOLD_DIRECTION["PS3000A_NONE"]
        extDirection = ps.PS3000A_THRESHOLD_DIRECTION["PS3000A_RISING"]
        auxDirection = ps.PS3000A_THRESHOLD_DIRECTION["PS3000A_NONE"]

        self.status["setTrigDir"] = ps.ps3000aSetTriggerChannelDirections(self.chandle, 
                                                                    channelADirection, 
                                                                    channelBDirection, 
                                                                    channelCDirection, 
                                                                    channelDDirection, 
                                                                    extDirection, 
                                                                    auxDirection)
        assert_pico_ok(self.status["setTrigDir"])

        # Setting the number of sample to be collected
        self.maxsamples = self.preTriggerSamples + self.postTriggerSamples

        wavetype = ctypes.c_int16(1)
        sweepType = ctypes.c_int32(0)
        triggertype = ctypes.c_int32(0)
        triggerSource = ctypes.c_int32(4)


        timeIntervalns = ctypes.c_float()
        returnedMaxSamples = ctypes.c_int16()
        self.status["GetTimebase"] = ps.ps3000aGetTimebase2(self.chandle, 
                                                    self.timebase, 
                                                    self.maxsamples, 
                                                    ctypes.byref(timeIntervalns), 
                                                    1, 
                                                    ctypes.byref(returnedMaxSamples), 
                                                    0)
        assert_pico_ok(self.status["GetTimebase"])

        # Creates a overlow location for data
        self.overflow = ctypes.c_int16()
        # Creates converted types maxsamples
        self.cmaxSamples = ctypes.c_int32(self.maxsamples)


        # Create buffers ready for assigning pointers for data collection
        global bufferAMax
        global bufferAMin
        bufferAMax = (ctypes.c_int16 * self.maxsamples)()
        bufferAMin = (ctypes.c_int16 * self.maxsamples)() # used for downsampling which isn't in the scope of this example

        # Setting the data buffer location for data collection from channel A
        # Handle = Chandle
        # source = ps3000A_channel_A = 0
        # Buffer max = ctypes.byref(bufferAMax)
        # Buffer min = ctypes.byref(bufferAMin)
        # Buffer length = maxsamples
        # Segment index = 0
        # Ratio mode = ps3000A_Ratio_Mode_None = 0
        self.status["SetDataBuffers"] = ps.ps3000aSetDataBuffers(self.chandle, 
                                                            0, 
                                                            ctypes.byref(bufferAMax), 
                                                            ctypes.byref(bufferAMin), 
                                                            self.maxsamples, 
                                                            0, 
                                                            0)



        # Begin streaming mode:
        self.sampleInterval = ctypes.c_int32(2)
        self.sampleUnits = ps.PS3000A_TIME_UNITS['PS3000A_US']
        self.autoStopOn = 0
        # No downsampling:
        downsampleRatio = 1
        self.totalSamples = self.numBuffersToCapture * self.maxsamples
        self.status["runStreaming"] = ps.ps3000aRunStreaming(self.chandle,
                                                        ctypes.byref(self.sampleInterval),
                                                        self.sampleUnits,
                                                        self.preTriggerSamples,
                                                        self.totalSamples,
                                                        self.autoStopOn,
                                                        downsampleRatio,
                                                        ps.PS3000A_RATIO_MODE['PS3000A_RATIO_MODE_NONE'],
                                                        self.maxsamples)
        assert_pico_ok(self.status["runStreaming"])

        self.actualSampleInterval = self.sampleInterval.value
        self.actualSampleIntervalNs = self.actualSampleInterval * 1000

        print("Capturing at sample interval %s ns" % self.actualSampleIntervalNs)


        # We need a big buffer, not registered with the driver, to keep our complete capture in.
        global bufferCompleteA, nextSample, autoStopOuter, wasCalledBack, current_call_no
        bufferCompleteA = np.ones(shape=self.maxsamples, dtype=np.int16)

        current_call_no = 0
        nextSample = 0
        autoStopOuter = False
        wasCalledBack = False
        self.cFuncPtr = ps.StreamingReadyType(self.streaming_callback)


    def streaming_callback(handle, noOfSamples, startIndex, overflow, triggerAt, triggered, autoStop, param,X):
        # print(f'handle:{handle}')
        # print(f'noOfSamples: {noOfSamples}')
        # print(f'startIndex: {startIndex}')
        # print(f'overflow: {overflow}')
        # print(f'triggeredAt: {triggerAt}')
        # print(f'triggered: {triggered}')
        print(f'autoStop: {autoStop}')
        # print(f'param: {param}')
        # print(f'mystery: {X}')



        global nextSample, autoStopOuter, wasCalledBack, bufferAMax, bufferAMin, bufferCompleteA, current_call_no
        wasCalledBack = True
        destEnd = nextSample + noOfSamples
        sourceEnd = startIndex + noOfSamples
        bufferCompleteA[nextSample:destEnd] = bufferAMax[startIndex:sourceEnd]
        print(np.max(bufferCompleteA))
        print(f'wasCalledBack: {wasCalledBack}')
        current_call_no += 1

        nextSample += noOfSamples

        if autoStop:
            autoStopOuter = True




    def update_plot(self):
        global bufferCompleteA, nextSample, autoStopOuter, wasCalledBack, current_call_no
        # Simulate new data

        self.i += 1
        # sos = signal.butter(10, 20, 'lp', fs=2025, output='sos')  
        # correction = 180
        # ## angle stuff is about 15ms
        current_angle = list(self.sensor.euler)

        # if None in current_angle:
        #     return()

        # if current_angle[2] < 0:
        #     correction *= -1

        # current_angle[2] = current_angle[2]-correction

        # # if abs(current_angle[0]) > 90:
        # #     # current_angle[0] = current_angle[0] - 360
        # #     # if current_angle[0] > 180:
        # #     current_angle[0] = previous_angle[0]

        # if abs(current_angle[1]) > 90:
        #     current_angle[1] = self.previous_angle[1]
        # if abs(current_angle[2]) > 90:
        #     current_angle[2] = self.previous_angle[2]

        # current_angle[0] = np.radians(current_angle[0])
        # self.pitch_angles.append(current_angle[1]*1.5)
        # self.roll_angles.append(current_angle[2]*1.5)
        # self.yaw_angles.append(current_angle[0])

        # if len(self.pitch_angles) > 10:
        #     self.pitch_angles.pop(0)

        # if len(self.roll_angles) > 10:
        #     self.roll_angles.pop(0)

        # if len(self.yaw_angles) > 10:
        #     self.yaw_angles.pop(0)

        # set_pitch = np.mean(self.pitch_angles)
        # set_roll = np.mean(self.roll_angles)
        # set_yaw = np.mean(self.yaw_angles)

        # pitch_mean.set_data([0,get_line_x(set_pitch)],[0,-1000])
        # roll_mean.set_data([0,get_line_x(set_roll)],[0,-1000])
        # yaw_mean.set_data([set_yaw,set_yaw],[1,0])
        # switch_angle = False
        

        # # setting data is about < 1ms
        # pitch.set_data([0,get_line_x(pitch_angles[-1])],[0,-1000])
        # roll.set_data([0,get_line_x(roll_angles[-1])],[0,-1000])
        # yaw.set_data([yaw_angles[-1],yaw_angles[-1]],[1,0])
        

        # output_samples = []

        # angle = self.sensor.euler
        # current_angle = list(angle)
        # if None in current_angle:
        #     return()

        print(f'before next sample: {nextSample}')

        while nextSample < self.totalSamples:
            wasCalledBack = False
            self.status["getStreamingLastestValues"] = ps.ps3000aGetStreamingLatestValues(self.chandle, self.cFuncPtr, None)
            print(f'nextSample:{nextSample}')
            print(f'totalSamples: {self.totalSamples}')
            if current_call_no > 25:
                print('reached 25! breaking out')
                break
            if not wasCalledBack:
                # If we weren't called back by the driver, this means no data is ready. Sleep for a short while before trying
                # again.
                print('sleeping')
                time.sleep(0.001)

        # ps.ClearTriggerReady(0)
        nextSample = 0
        autoStopOuter = False
        print(f'After next sample: {nextSample}')

        print("Done grabbing values.")

        # Find maximum ADC count value
        # handle = chandle
        # pointer to value = ctypes.byref(maxADC)
        maxADC = ctypes.c_int16()
        self.status["maximumValue"] = ps.ps3000aMaximumValue(self.chandle, ctypes.byref(maxADC))
        assert_pico_ok(self.status["maximumValue"])

        # Convert ADC counts data to mV
        print(np.max(bufferCompleteA))
        adc2mVChAMax =  adc2mV(bufferCompleteA, self.chARange, self.maxADC)
        print(np.max(adc2mVChAMax[0:10]))
        # Create time data

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
        # try:
        #     filtered_samples = signal.sosfilt(sos, adc2mVChAMax)
        # except ValueError:
        #     return()


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
        distance = np.arange(len(adc2mVChAMax))/(4*1000)*13/10
        refresh_rate = time.time() - self.t_start
        tx = f'Refresh rate (s): {refresh_rate}'
        print(tx)
        self.text_item.setText(tx)

        self.t_start = time.time()
        # self.text_item.setPos(0, max(self.y_data) if self.y_data else 0)

        print(f'len of filtered_samples: {len(adc2mVChAMax)}')
        self.x_data = distance[0:5000]
        self.y_data = adc2mVChAMax[0:5000]
        
        # Update plot
        self.plot.setData(self.x_data, self.y_data)

    def run(self):
        sys.exit(self.app.exec())

if __name__ == '__main__':
    plotter = TimeSeriesPlotter()
    plotter.run()
