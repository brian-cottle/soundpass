import sys
import numpy as np
from PyQt6.QtWidgets import QApplication
from PyQt6 import QtWidgets
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
from PyQt6.QtCore import Qt
import os
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as R
from skimage.filters import gaussian
import matplotlib
import matplotlib.cm


# Define a simple 1D segmentation network
class SegmentationNetwork(nn.Module):
    def __init__(self, input_channels, num_classes,kernel_size=45,hidden_channels=85):
        super(SegmentationNetwork, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, hidden_channels, kernel_size=kernel_size, padding='same')
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_channels, num_classes, kernel_size=kernel_size, padding='same')
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return self.softmax(x)


def get_line_x(angle):
    end_x = 1000*math.tan(np.radians(abs(angle)))

    end_x *= math.copysign(1,angle)

    return(end_x)


def setup_IMU():
    i2c = busio.I2C(board.SCL, board.SDA)
    sensor = adafruit_bno055.BNO055_I2C(i2c)
    sensor.axis_remap = (0,1,2,0,1,1)
    return(sensor)



class TimeSeriesPlotter:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.plot_widget = pg.PlotWidget()
        self.w = QtWidgets.QWidget()
        self.w.setWindowTitle('PyQtGraph example')
        layout = QtWidgets.QGridLayout()
        self.w.setLayout(layout)
        self.btn = QtWidgets.QPushButton('Toggle Data Acquisition')
        self.btn.setFixedSize(400, 200)  # Set button size
        self.btn.setStyleSheet("background-color: #4CAF50; color: white; font-size: 25px; border-radius: 10px; padding: 10px;")
        
        layout.addWidget(self.btn, 0, 0)  # button goes in upper-left
        self.segmentation_btn = QtWidgets.QPushButton('Toggle Segmentation')
        self.segmentation_btn.setFixedSize(400, 200)  # Set button size
        self.segmentation_btn.setStyleSheet("background-color: #2196F3; color: white; font-size: 25px; border-radius: 10px; padding: 10px;")

        layout.addWidget(self.segmentation_btn, 1, 0)  # button goes below the first button

        layout.addWidget(self.plot_widget, 0, 2, 10, 1)

        self.plot_widget.getPlotItem().getAxis('bottom').setLabel('Distance (cm)', **{'font-size': '25px'})
        axis = self.plot_widget.getPlotItem().getAxis('bottom')
        axis.setStyle(tickFont=pg.QtGui.QFont('Arial', 45))
        axis.setPen(pg.mkPen(color='w'))  # Set the font color to white

        # Add a polar-like plot to the right of the main plot
        self.polar_plot = pg.PlotWidget()
        self.polar_plot.setAspectLocked(True)  # Ensure the plot is square
        self.polar_plot.hideAxis('left')
        self.polar_plot.hideAxis('bottom')
        self.polar_plot.setBackground('k')  # Set background to black
        layout.addWidget(self.polar_plot, 0, 4, 10, 1)  # Add to the layout next to the main plot
        self.image_dimensions = 100
        
        self.bg_image_array = np.zeros((self.image_dimensions, self.image_dimensions), dtype=float)
        self.blurred_image_array = gaussian(self.bg_image_array, sigma=2)
        self.bg_image = pg.ImageItem(self.bg_image_array)
        # self.bg_image.setLevels([0, 1])  # Set display range for greyscale
        
        # Optionally scale/position the image to fit your plot coordinates
        self.bg_image.setRect(pg.QtCore.QRectF(-1, -1, 2, 2))  # Example: fill from -1 to 1 in both axes

        # Add the image to the polar plot (add before other items for background)
        self.polar_plot.addItem(self.bg_image)


        # Add a circular grid to the polar plot
        self.polar_grid = pg.ScatterPlotItem(
            x=[np.cos(np.radians(angle)) for angle in range(0, 360, 30)],
            y=[np.sin(np.radians(angle)) for angle in range(0, 360, 30)],
            pen=pg.mkPen('w'),
            brush=pg.mkBrush(255, 255, 255, 50),
            size=5
        )
        self.polar_plot.addItem(self.polar_grid)


        # Add toggle for colormap/greyscale
        self.colormap_toggle_btn = QtWidgets.QPushButton('Toggle Colormap')
        self.colormap_toggle_btn.setFixedSize(400, 100)
        self.colormap_toggle_btn.setStyleSheet("background-color: #9C27B0; color: white; font-size: 25px; border-radius: 10px; padding: 10px;")
        layout.addWidget(self.colormap_toggle_btn, 3, 0)  # Add below model weights selector
        self.use_colormap = True  # Track colormap state



        # # Add a line to represent the angle
        # self.angle_line = pg.PlotDataItem(pen=pg.mkPen('r', width=3))
        # self.polar_plot.addItem(self.angle_line)

        # # Add lines for roll, pitch, and yaw deviations
        # self.roll_line = pg.PlotDataItem(pen=pg.mkPen('r', width=3))   # Red for roll
        # self.pitch_line = pg.PlotDataItem(pen=pg.mkPen('g', width=3))  # Green for pitch
        # self.yaw_line = pg.PlotDataItem(pen=pg.mkPen('b', width=3))    # Blue for yaw
        # self.polar_plot.addItem(self.roll_line)
        # self.polar_plot.addItem(self.pitch_line)
        # self.polar_plot.addItem(self.yaw_line)


        # Add cross-hairs for tip position
        self.crosshair_v = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('y', width=2))
        self.crosshair_h = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('y', width=2))
        self.polar_plot.addItem(self.crosshair_v)
        self.polar_plot.addItem(self.crosshair_h)
        self.crosshair_center = (0, 0)  # Start at center


        # Add static cross-hairs at (0,0) (grey, semi-transparent, dotted)
        static_pen = pg.mkPen(color=(150, 150, 150, 120), width=2, style=Qt.PenStyle.DotLine)
        self.static_crosshair_v = pg.InfiniteLine(angle=90, movable=False, pen=static_pen)
        self.static_crosshair_h = pg.InfiniteLine(angle=0, movable=False, pen=static_pen)
        self.polar_plot.addItem(self.static_crosshair_v)
        self.polar_plot.addItem(self.static_crosshair_h)
        self.static_crosshair_v.setPos(0)
        self.static_crosshair_h.setPos(0)

        # Initialize angle data
        self.current_angle = 0

        # Add a button to toggle angle display on the polar plot
        self.toggle_angle_btn = QtWidgets.QPushButton('Show Angle')
        self.toggle_angle_btn.setFixedSize(400, 200)
        self.toggle_angle_btn.setStyleSheet("background-color: #FF9800; color: white; font-size: 25px; border-radius: 10px; padding: 10px;")
        layout.addWidget(self.toggle_angle_btn, 2, 0)  # Place below the polar plot

        self.show_angle = False  # Track angle display state


        # self.plot_widget.show()
        self.w.show()
        self.plot1 = self.plot_widget.plot(pen='y')
        self.plot1.setAlpha(0.4, False)
        self.plot_widget.setMouseEnabled(x=False, y=False)

        self.x_data = []
        self.y_data = []
        self.y2_data = []
    
        # Add a text box for annotations
        self.annotation_input = QtWidgets.QLineEdit()
        self.annotation_input.setPlaceholderText("Enter annotation here...")
        self.annotation_input.setFixedSize(400, 100)  # Set text box size
        self.annotation_input.setStyleSheet("font-size: 35px; padding: 10px;")
        layout.addWidget(self.annotation_input, 4, 0)  # text box goes below the buttons
        self.annotation_text = ''

        # Add a text box to display the refresh rate
        self.refresh_rate_display = QtWidgets.QLineEdit()
        self.refresh_rate_display.setReadOnly(True)  # Make it read-only
        self.refresh_rate_display.setFixedSize(400, 100)  # Set text box size
        self.refresh_rate_display.setStyleSheet("font-size: 35px; padding: 10px;")
        layout.addWidget(self.refresh_rate_display, 5, 0)  # text box goes below the annotation input

        # Add a text box to display the data acquisition count
        self.data_acquisition_count_display = QtWidgets.QLineEdit()
        self.data_acquisition_count_display.setReadOnly(True)  # Make it read-only
        self.data_acquisition_count_display.setFixedSize(400, 100)  # Set text box size
        self.data_acquisition_count_display.setStyleSheet("font-size: 35px; padding: 10px;")
        self.data_acquisition_count_display.setText("Sample Number: 0")
        layout.addWidget(self.data_acquisition_count_display, 6, 0)  # Add below the segmentation count display

        # Add a text box to display the elapsed time for segmentation
        self.elapsed_time_display = QtWidgets.QLineEdit()
        self.elapsed_time_display.setReadOnly(True)  # Make it read-only
        self.elapsed_time_display.setFixedSize(400, 100)  # Set text box size
        self.elapsed_time_display.setStyleSheet("font-size: 35px; padding: 10px;")
        self.elapsed_time_display.setText("Elapsed Time: 0m 0s")
        layout.addWidget(self.elapsed_time_display, 7, 0)  # Add below the data acquisition count display

        # Add a spin box for segmentation threshold
        self.threshold_spinbox = QtWidgets.QDoubleSpinBox()
        self.threshold_spinbox.setDecimals(1)  # Allow one decimal place
        self.threshold_spinbox.setSingleStep(0.1)  # Step size of 0.1
        self.threshold_spinbox.setRange(0.0, 1.0)  # Set the range for the threshold
        self.threshold_spinbox.setValue(0.5)  # Set the default value
        self.threshold_spinbox.setFixedSize(400, 100)  # Set size
        self.threshold_spinbox.setStyleSheet(
            "font-size: 35px; padding: 10px; "
            "QAbstractSpinBox::up-arrow { width: 50px; height: 50px; } "
            "QAbstractSpinBox::down-arrow { width: 50px; height: 50px; }"
        )
        layout.addWidget(self.threshold_spinbox, 8, 0)  # Add below the elapsed time display


        # Create an instance of the segmentation model
        self.segmentation_model = SegmentationNetwork(input_channels=1, num_classes=2,)

        # --- Model weights selection dropdown ---
        self.model_weights_files = [
            '/home/briancottle/Code/SoundPass/Live_Models/cadaver_model_v02_10.pth',
            '/home/briancottle/Code/SoundPass/Live_Models/cadaver_model_v01_49.pth',
            '/home/briancottle/Code/SoundPass/Live_Models/phantom_model_V10_17.pth',
            # Add more model weight file paths here as needed
        ]
        self.model_weights_selector = QtWidgets.QComboBox()
        self.model_weights_selector.addItems([os.path.basename(f) for f in self.model_weights_files])
        self.model_weights_selector.setFixedSize(400, 100)
        self.model_weights_selector.setStyleSheet("font-size: 25px; padding: 10px;")
        layout.addWidget(self.model_weights_selector, 9, 0)  # Place below the threshold spinbox

        self.model_weights_selector.currentIndexChanged.connect(self.on_model_weights_changed)
        self.model_weights_selector.setCurrentIndex(0)  # Set default selection to the first item

        weights_file = self.model_weights_files[0]  # Default to the first file

        # Load the weights from a file
        self.segmentation_model.load_state_dict(torch.load(weights_file))

        # Set the model to evaluation mode
        self.segmentation_model.eval()

        # Set a default threshold value
        self.segmentation_threshold = 0.5

        # Track data acquisition count and session start time
        self.data_acquisition_count = 0
        self.data_acquisition_start_time = None

        self.pitch_angles = [0]
        self.roll_angles = [0]
        self.yaw_angles = [0]
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(1) # update at least every 1 ms
        self.i = 0
        self.text_item = pg.TextItem(text="Initial Text", anchor=(5, 100), color=(255, 255, 0),html='<span style="font-size: 16pt;">Initial Text</span>',)
        self.plot_widget.setYRange(-10000,10000)
        self.preTriggerSamples = 0 # Hz
        self.postTriggerSamples = 30000 # Hz
        self.timebase = 2 # 2ns per sample
        self.nChannelProperties = 1
        self.autoTriggerMilliseconds = 10000 # 10s before auto triggering
        self.chARange = 6 # 1V
        self.advanced_trigger_threshold = 500 # mV
        self.sig_gen_peak_to_peak = 2000000 # 2V
        self.sig_gen_hz = 1000
        self.sig_gen_offset = 00000000 # 0V

        self.last_time = 0
        self.t_start = time.time()
        self.distance = np.arange((self.preTriggerSamples + self.postTriggerSamples))/(4*1000)*13/10
        self.x_data = self.distance
        self.save_data = False
        self.status = {}
        self.chandle = ctypes.c_int16()
        self.segmentation_enabled = False

        self.past_segmentations = []

        self.qt_connections()
        self.sensor = setup_IMU()
        



        # Create a directory to save files using the current date and time
        now = datetime.now()
        self.save_directory = "/home/briancottle/Code/SoundPass/new_datasets/" + now.strftime("%Y-%m-%d")
        os.makedirs(self.save_directory, exist_ok=True)


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


        # Create buffers ready for assigning pointers for data collection
        self.bufferAMax = (ctypes.c_int16 * self.maxsamples)()
        self.bufferAMin = (ctypes.c_int16 * self.maxsamples)() # used for downsampling which isn't in the scope of this example

        # Setting the data buffer location for data collection from channel A
        # Handle = Chandle
        # source = ps3000A_channel_A = 0
        # Buffer max = ctypes.byref(bufferAMax)
        # Buffer min = ctypes.byref(bufferAMin)
        # Buffer length = maxsamples
        # Segment index = 0
        # Ratio mode = ps3000A_Ratio_Mode_None = 0


        # ----------------------------------------------------
        # self.status["SetDataBuffers"] = ps.ps3000aSetDataBuffers(self.chandle, 
        #                                                     0, 
        #                                                     ctypes.byref(self.bufferAMax), 
        #                                                     ctypes.byref(self.bufferAMin), 
        #                                                     self.maxsamples, 
        #                                                     0, 
        #                                                     0)

        # assert_pico_ok(self.status["SetDataBuffers"])

        # # Creates a overlow location for data
        self.overflow = (ctypes.c_int16 * 20)()
        # Creates converted types maxsamples
        self.cmaxSamples = ctypes.c_int32(self.maxsamples)
        # -----------------------------------------------------

        self.status["MemorySegments"] = ps.ps3000aMemorySegments(self.chandle, 20, ctypes.byref(self.cmaxSamples))
        assert_pico_ok(self.status["MemorySegments"])

        # sets number of captures
        self.status["SetNoOfCaptures"] = ps.ps3000aSetNoOfCaptures(self.chandle, 20)
        assert_pico_ok(self.status["SetNoOfCaptures"])

        # Starts the block capture
        # Handle = chandle
        # Number of prTriggerSamples
        # Number of postTriggerSamples
        # Timebase = 2 = 4ns (see Programmer's guide for more information on timebases)
        # time indisposed ms = None (This is not needed within the example)
        # Segment index = 0
        # LpRead = None


        # Create buffers ready for assigning pointers for data collection
        #bufferAMax = (ctypes.c_int16 * maxsamples)()
        #bufferAMin = (ctypes.c_int16 * maxsamples)() # used for downsampling which isn't in the scope of this example

        self.bufferAMax = np.empty(self.maxsamples, dtype=np.dtype('int16'))
        self.bufferAMin = np.empty(self.maxsamples, dtype=np.dtype('int16')) # used for downsampling which isn't in the scope of this example

        self.status["SetDataBuffers"] = ps.ps3000aSetDataBuffers(self.chandle, 0, self.bufferAMax.ctypes.data, self.bufferAMin.ctypes.data, self.maxsamples, 0, 0)
        assert_pico_ok(self.status["SetDataBuffers"])


        # Create buffers ready for assigning pointers for data collection
        self.bufferAMax1 = np.empty(self.maxsamples, dtype=np.dtype('int16'))
        self.bufferAMin1 = np.empty(self.maxsamples, dtype=np.dtype('int16')) # used for downsampling which isn't in the scope of this example

        self.status["SetDataBuffers"] = ps.ps3000aSetDataBuffers(self.chandle, 0, self.bufferAMax1.ctypes.data, self.bufferAMin1.ctypes.data, self.maxsamples, 1, 0)
        assert_pico_ok(self.status["SetDataBuffers"])


        # Create buffers ready for assigning pointers for data collection
        self.bufferAMax2 = np.empty(self.maxsamples, dtype=np.dtype('int16'))
        self.bufferAMin2 = np.empty(self.maxsamples, dtype=np.dtype('int16')) # used for downsampling which isn't in the scope of this example

        self.status["SetDataBuffers"] = ps.ps3000aSetDataBuffers(self.chandle, 0, self.bufferAMax2.ctypes.data, self.bufferAMin2.ctypes.data, self.maxsamples, 2, 0)
        assert_pico_ok(self.status["SetDataBuffers"])


        # Create buffers ready for assigning pointers for data collection
        self.bufferAMax3 = np.empty(self.maxsamples, dtype=np.dtype('int16'))
        self.bufferAMin3 =np.empty(self.maxsamples, dtype=np.dtype('int16')) # used for downsampling which isn't in the scope of this example

        self.status["SetDataBuffers"] = ps.ps3000aSetDataBuffers(self.chandle, 0, self.bufferAMax3.ctypes.data, self.bufferAMin3.ctypes.data, self.maxsamples, 3, 0)
        assert_pico_ok(self.status["SetDataBuffers"])


        # Create buffers ready for assigning pointers for data collection
        self.bufferAMax4 = np.empty(self.maxsamples, dtype=np.dtype('int16'))
        self.bufferAMin4 = np.empty(self.maxsamples, dtype=np.dtype('int16')) # used for downsampling which isn't in the scope of this example

        self.status["SetDataBuffers"] = ps.ps3000aSetDataBuffers(self.chandle, 0, self.bufferAMax4.ctypes.data, self.bufferAMin4.ctypes.data, self.maxsamples, 4, 0)
        assert_pico_ok(self.status["SetDataBuffers"])


        # Create buffers ready for assigning pointers for data collection
        self.bufferAMax5 = np.empty(self.maxsamples, dtype=np.dtype('int16'))
        self.bufferAMin5 = np.empty(self.maxsamples, dtype=np.dtype('int16')) # used for downsampling which isn't in the scope of this example

        self.status["SetDataBuffers"] = ps.ps3000aSetDataBuffers(self.chandle, 0, self.bufferAMax5.ctypes.data, self.bufferAMin5.ctypes.data, self.maxsamples, 5, 0)
        assert_pico_ok(self.status["SetDataBuffers"])


        # Create buffers ready for assigning pointers for data collection
        self.bufferAMax6 = np.empty(self.maxsamples, dtype=np.dtype('int16'))
        self.bufferAMin6 = np.empty(self.maxsamples, dtype=np.dtype('int16')) # used for downsampling which isn't in the scope of this example

        self.status["SetDataBuffers"] = ps.ps3000aSetDataBuffers(self.chandle, 0, self.bufferAMax6.ctypes.data, self.bufferAMin6.ctypes.data, self.maxsamples, 6, 0)
        assert_pico_ok(self.status["SetDataBuffers"])


        # Create buffers ready for assigning pointers for data collection
        self.bufferAMax7 = np.empty(self.maxsamples, dtype=np.dtype('int16'))
        self.bufferAMin7 = np.empty(self.maxsamples, dtype=np.dtype('int16')) # used for downsampling which isn't in the scope of this example

        self.status["SetDataBuffers"] = ps.ps3000aSetDataBuffers(self.chandle, 0, self.bufferAMax7.ctypes.data, self.bufferAMin7.ctypes.data, self.maxsamples, 7, 0)
        assert_pico_ok(self.status["SetDataBuffers"])


        # Create buffers ready for assigning pointers for data collection
        self.bufferAMax8 = np.empty(self.maxsamples, dtype=np.dtype('int16'))
        self.bufferAMin8 = np.empty(self.maxsamples, dtype=np.dtype('int16')) # used for downsampling which isn't in the scope of this example

        self.status["SetDataBuffers"] = ps.ps3000aSetDataBuffers(self.chandle, 0, self.bufferAMax8.ctypes.data, self.bufferAMin8.ctypes.data, self.maxsamples, 8, 0)
        assert_pico_ok(self.status["SetDataBuffers"])


        # Create buffers ready for assigning pointers for data collection
        self.bufferAMax9 = np.empty(self.maxsamples, dtype=np.dtype('int16'))
        self.bufferAMin9 = np.empty(self.maxsamples, dtype=np.dtype('int16')) # used for downsampling which isn't in the scope of this example

        self.status["SetDataBuffers"] = ps.ps3000aSetDataBuffers(self.chandle, 0, self.bufferAMax9.ctypes.data, self.bufferAMin9.ctypes.data, self.maxsamples, 9, 0)
        assert_pico_ok(self.status["SetDataBuffers"])


        # --------
        # Create buffers ready for assigning pointers for data collection
        self.bufferAMax10 = np.empty(self.maxsamples, dtype=np.dtype('int16'))
        self.bufferAMin10 = np.empty(self.maxsamples, dtype=np.dtype('int16')) # used for downsampling which isn't in the scope of this example

        self.status["SetDataBuffers"] = ps.ps3000aSetDataBuffers(self.chandle, 0, self.bufferAMax10.ctypes.data, self.bufferAMin10.ctypes.data, self.maxsamples, 10, 0)
        assert_pico_ok(self.status["SetDataBuffers"])


        # Create buffers ready for assigning pointers for data collection
        self.bufferAMax11 = np.empty(self.maxsamples, dtype=np.dtype('int16'))
        self.bufferAMin11 = np.empty(self.maxsamples, dtype=np.dtype('int16')) # used for downsampling which isn't in the scope of this example

        self.status["SetDataBuffers"] = ps.ps3000aSetDataBuffers(self.chandle, 0, self.bufferAMax11.ctypes.data, self.bufferAMin11.ctypes.data, self.maxsamples, 11, 0)
        assert_pico_ok(self.status["SetDataBuffers"])


        # Create buffers ready for assigning pointers for data collection
        self.bufferAMax12 = np.empty(self.maxsamples, dtype=np.dtype('int16'))
        self.bufferAMin12 = np.empty(self.maxsamples, dtype=np.dtype('int16')) # used for downsampling which isn't in the scope of this example

        self.status["SetDataBuffers"] = ps.ps3000aSetDataBuffers(self.chandle, 0, self.bufferAMax12.ctypes.data, self.bufferAMin12.ctypes.data, self.maxsamples, 12, 0)
        assert_pico_ok(self.status["SetDataBuffers"])


        # Create buffers ready for assigning pointers for data collection
        self.bufferAMax13 = np.empty(self.maxsamples, dtype=np.dtype('int16'))
        self.bufferAMin13 = np.empty(self.maxsamples, dtype=np.dtype('int16')) # used for downsampling which isn't in the scope of this example

        self.status["SetDataBuffers"] = ps.ps3000aSetDataBuffers(self.chandle, 0, self.bufferAMax13.ctypes.data, self.bufferAMin13.ctypes.data, self.maxsamples, 13, 0)
        assert_pico_ok(self.status["SetDataBuffers"])


        # Create buffers ready for assigning pointers for data collection
        self.bufferAMax14 = np.empty(self.maxsamples, dtype=np.dtype('int16'))
        self.bufferAMin14 = np.empty(self.maxsamples, dtype=np.dtype('int16')) # used for downsampling which isn't in the scope of this example

        self.status["SetDataBuffers"] = ps.ps3000aSetDataBuffers(self.chandle, 0, self.bufferAMax14.ctypes.data, self.bufferAMin14.ctypes.data, self.maxsamples, 14, 0)
        assert_pico_ok(self.status["SetDataBuffers"])


        # Create buffers ready for assigning pointers for data collection
        self.bufferAMax15 = np.empty(self.maxsamples, dtype=np.dtype('int16'))
        self.bufferAMin15 = np.empty(self.maxsamples, dtype=np.dtype('int16')) # used for downsampling which isn't in the scope of this example

        self.status["SetDataBuffers"] = ps.ps3000aSetDataBuffers(self.chandle, 0, self.bufferAMax15.ctypes.data, self.bufferAMin15.ctypes.data, self.maxsamples, 15, 0)
        assert_pico_ok(self.status["SetDataBuffers"])


        # Create buffers ready for assigning pointers for data collection
        self.bufferAMax16 = np.empty(self.maxsamples, dtype=np.dtype('int16'))
        self.bufferAMin16 = np.empty(self.maxsamples, dtype=np.dtype('int16')) # used for downsampling which isn't in the scope of this example

        self.status["SetDataBuffers"] = ps.ps3000aSetDataBuffers(self.chandle, 0, self.bufferAMax16.ctypes.data, self.bufferAMin16.ctypes.data, self.maxsamples, 16, 0)
        assert_pico_ok(self.status["SetDataBuffers"])


        # Create buffers ready for assigning pointers for data collection
        self.bufferAMax17 = np.empty(self.maxsamples, dtype=np.dtype('int16'))
        self.bufferAMin17 = np.empty(self.maxsamples, dtype=np.dtype('int16')) # used for downsampling which isn't in the scope of this example

        self.status["SetDataBuffers"] = ps.ps3000aSetDataBuffers(self.chandle, 0, self.bufferAMax17.ctypes.data, self.bufferAMin17.ctypes.data, self.maxsamples, 17, 0)
        assert_pico_ok(self.status["SetDataBuffers"])


        # Create buffers ready for assigning pointers for data collection
        self.bufferAMax18 = np.empty(self.maxsamples, dtype=np.dtype('int16'))
        self.bufferAMin18 = np.empty(self.maxsamples, dtype=np.dtype('int16')) # used for downsampling which isn't in the scope of this example

        self.status["SetDataBuffers"] = ps.ps3000aSetDataBuffers(self.chandle, 0, self.bufferAMax18.ctypes.data, self.bufferAMin18.ctypes.data, self.maxsamples, 18, 0)
        assert_pico_ok(self.status["SetDataBuffers"])


        # Create buffers ready for assigning pointers for data collection
        self.bufferAMax19 = np.empty(self.maxsamples, dtype=np.dtype('int16'))
        self.bufferAMin19 = np.empty(self.maxsamples, dtype=np.dtype('int16')) # used for downsampling which isn't in the scope of this example

        self.status["SetDataBuffers"] = ps.ps3000aSetDataBuffers(self.chandle, 0, self.bufferAMax19.ctypes.data, self.bufferAMin19.ctypes.data, self.maxsamples, 19, 0)
        assert_pico_ok(self.status["SetDataBuffers"])


    def keyPressEvent(self, event): # Checks if a specific key was pressed

        if event.key() == Qt.Key.Key_Escape:
            print("Escape key was pressed.")
            sys.exit(self.app.exec())
        elif event.key() == Qt.Key.Key_Space:
            print("Space bar was pressed.")
        else:
            # The 'event.text()' method retrieves the character or character
            # associated with the key press, and then prints it to the console.
            print(f"Key pressed: {event.text()}")


    def update_plot(self):
        # Simulate new data
        self.i += 1

        self.status["runblock"] = ps.ps3000aRunBlock(self.chandle, 
                                                    self.preTriggerSamples, 
                                                    self.postTriggerSamples, 
                                                    self.timebase, 
                                                    1, 
                                                    None, 
                                                    0, 
                                                    None, 
                                                    None)
        assert_pico_ok(self.status["runblock"])

        # Checks data collection to finish the capture
        ready = ctypes.c_int16(0)
        check = ctypes.c_int16(0)
        while ready.value == check.value:
            self.status["isReady"] = ps.ps3000aIsReady(self.chandle, ctypes.byref(ready))

        self.status["GetValuesBulk"] = ps.ps3000aGetValuesBulk(self.chandle, ctypes.byref(self.cmaxSamples), 0, 9, 1, 0, ctypes.byref(self.overflow))
        assert_pico_ok(self.status["GetValuesBulk"])

        all_samples = np.asarray([self.bufferAMax,
                                   self.bufferAMax1,
                                   self.bufferAMax2,
                                   self.bufferAMax3,
                                   self.bufferAMax4,
                                   self.bufferAMax5,
                                   self.bufferAMax6,
                                   self.bufferAMax7,
                                   self.bufferAMax8,
                                   self.bufferAMax9,
                                   self.bufferAMax10,
                                   self.bufferAMax11,
                                   self.bufferAMax12,
                                   self.bufferAMax13,
                                   self.bufferAMax14,
                                   self.bufferAMax15,
                                   self.bufferAMax16,
                                   self.bufferAMax17,
                                   self.bufferAMax18,
                                   self.bufferAMax19,
                                   ])

        average_samples = np.mean(all_samples, axis=0)
        # baselined_samples = np.asarray(average_samples - self.baseline_array)

        if self.segmentation_enabled:
            # Reshape the data to match the input shape of the model
            # baselined_samples = (baselined_samples - np.mean(baselined_samples)) / np.std(baselined_samples)
            down_sampled = average_samples[::30]
            input_data = torch.tensor(down_sampled).float().unsqueeze(0).unsqueeze(0)
            # Perform segmentation
            with torch.no_grad():
                output = self.segmentation_model(input_data)
            # Get binary prediction based on a threshold for the positive class
            positive_class_probabilities = output[:, 1, :].squeeze().numpy()
            # Use the user-defined threshold
            threshold = self.segmentation_threshold
            binary_prediction = (positive_class_probabilities > threshold).astype(int)
            predicted_class = binary_prediction  # Assign the binary prediction to predicted_class
            # predicted_class = torch.argmax(output, dim=1).squeeze().numpy()

            # Store the current segmentation in past_segmentations
            self.past_segmentations.append(predicted_class)

            # Ensure only the last 10 segmentations are kept
            if len(self.past_segmentations) > 10:
                self.past_segmentations.pop(0)

            # Compute the average segmentation
            averaged_segmentation = np.mean(self.past_segmentations, axis=0)

            # Reshape predicted_class to match the size of baselined_samples
            reshaped_predicted_class = np.interp(
                np.arange(len(average_samples)),
                np.linspace(0, len(average_samples) - 1, len(averaged_segmentation)),
                averaged_segmentation,
            ).round().astype(int)

            # Zero out segmentation in the first 10% of the signal
            first_10_percent = int(len(reshaped_predicted_class) * 0.15)
            reshaped_predicted_class[:first_10_percent] = 0

            # Update the region based on the predicted class
            if hasattr(self, 'highlighted_regions'):
                for region in self.highlighted_regions:
                    self.plot_widget.removeItem(region)
            self.highlighted_regions = []

            if reshaped_predicted_class.any():
                segment_indices = np.where(reshaped_predicted_class == 1)[0]
                if len(segment_indices) > 0:
                    # Find contiguous regions of positive values
                    regions = np.split(segment_indices, np.where(np.diff(segment_indices) != 1)[0] + 1)
                    for region in regions:
                        start_index = region[0]
                        end_index = region[-1]

                        # # Map the x-coordinates to indices of the data array
                        # start_index = max(0, np.searchsorted(self.distance, region[0]))
                        # end_index = min(len(average_samples), np.searchsorted(self.distance, region[-1]))


                        start_pos = self.x_data[start_index]
                        end_pos = self.x_data[end_index]
                        # print(f"Adding region from {start_pos} to {end_pos}")
                        # Create a new region for each contiguous segment
                        new_region = pg.LinearRegionItem([start_pos, end_pos])
                        new_region.setZValue(10)  # Ensure it appears above the plot
                        new_region.setBrush(pg.mkBrush(255, 0, 0, 50))  # Semi-transparent red
                        self.plot_widget.addItem(new_region)
                        self.highlighted_regions.append(new_region)


        refresh_rate = 1 / (time.time() - self.t_start)
        tx = f'Refresh rate (Hz): {refresh_rate:0.3f}'

        self.t_start = time.time()
        self.refresh_rate_display.setText(tx)

        self.y1_data = average_samples
        self.plot1.setData(self.x_data, self.y1_data)

        # Update elapsed time if data acquisition is enabled
        if self.save_data and self.data_acquisition_start_time is not None:
            elapsed_time = time.time() - self.data_acquisition_start_time
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            self.elapsed_time_display.setText(f"Elapsed Time: {minutes}m {seconds}s")


        if self.show_angle:
            # Get current orientation as quaternion from the sensor
            try:
                current_quat = R.from_quat(self.sensor.quaternion)  # [x, y, z, w] order
            except ValueError:
                print("Error: Quaternion data not available from the sensor.")
                self.sensor = setup_IMU()
                self.initial_quat = R.from_quat(self.sensor.quaternion)  # Reset initial quaternion

            # Compute relative rotation (from initial to current)
            relative_quat = current_quat * self.initial_quat.inv()

            # Get relative Euler angles (in device's frame)
            relative_euler = relative_quat.as_euler('xyz', degrees=True)
            relative_roll = relative_euler[0]
            relative_pitch = relative_euler[1]  # "Up/Down" relative to initial
            relative_yaw = relative_euler[2]    # "Left/Right" relative to initial

            # print(f"Relative Roll: {relative_roll:.2f}°, Relative Pitch: {relative_pitch:.2f}°, Relative Yaw: {relative_yaw:.2f}°")

            # Check for sudden changes in pitch or yaw
            if abs(relative_pitch - self.pitch_angles[-1]) > 45 or abs(relative_yaw - self.yaw_angles[-1]) > 45:
                print(f"Sudden change detected. Resetting IMU... Pitch Change: {relative_pitch - self.pitch_angles[-1]:.2f}°, Yaw Change: {relative_yaw - self.yaw_angles[-1]:.2f}°")
                print(f"Recent Pitch Values: {self.pitch_angles[-3:]}, Newest: {relative_pitch}")
                print(f"Recent Yaw Values: {self.yaw_angles[-3:]}, Newest: {relative_yaw}")
                self.sensor = setup_IMU()
                self.initial_quat = R.from_quat(self.sensor.quaternion)  # Reset initial quaternion
            else:
                # Update pitch, roll, and yaw angles
                self.pitch_angles.append(relative_pitch)
                self.roll_angles.append(relative_roll)
                self.yaw_angles.append(relative_yaw)

                if len(self.pitch_angles) > 10:
                    self.pitch_angles.pop(0)
                    self.roll_angles.pop(0)
                    self.yaw_angles.pop(0)


            d = 2
            pitch_rad = np.radians(relative_pitch)
            yaw_rad = np.radians(relative_yaw)

            y = d * np.sin(yaw_rad) * np.cos(pitch_rad)
            x = d * np.sin(pitch_rad)
            z = d * np.cos(yaw_rad) * np.cos(pitch_rad)

            self.crosshair_v.setPos(-x)
            self.crosshair_h.setPos(-y)

            ix, iy = self.plot_coords_to_image_indices(-y, -x) # swapping x and y
            if reshaped_predicted_class.any():
                # Find the closest positive segmentation value to zero
                positive_indices = np.where(reshaped_predicted_class == 1)[0]
                if len(positive_indices) > 0:

                    # Weight the value based on the amplitude of the signal at the segmentation
                    amplitude_weight = np.mean(np.abs(average_samples[reshaped_predicted_class == 1]))

                    closest_index = positive_indices[np.argmin(np.abs(self.x_data[positive_indices]))]
                    closest_distance = np.abs(self.x_data[closest_index])
                    # Normalize the distance to a value between 0 and 1
                    normalized_value = 1.0 - (closest_distance / np.max(np.abs(self.x_data)))
                    # Apply weighting to the normalized value
                    weighted_value = normalized_value ** 2  # Squaring emphasizes closer values and diminishes smaller ones
                    # weighted_value *= amplitude_weight
                    self.bg_image_array[iy, ix] = weighted_value
                    # self.bg_image_array[iy, ix] = normalized_value

                    self.blurred_image_array = gaussian(self.bg_image_array, sigma=2)



            if self.use_colormap:
                self.colored_img = self.apply_colormap(self.blurred_image_array, cmap_name='jet')
                self.bg_image.setImage(self.colored_img, autoLevels=True)
            else:
                self.bg_image.setImage(self.blurred_image_array, autoLevels=True)

            # self.colored_img = self.apply_colormap(self.blurred_image_array, cmap_name='jet')
            # print(np.unique(self.colored_img))
            # self.bg_image.setImage(self.colored_img, autoLevels=False)

            # self.bg_image.setImage(np.random.rand(self.image_dimensions, self.image_dimensions))



        if self.save_data:
            now = datetime.now()
            formatted_date_time = int(time.time()*1000)
            segmentation_number = len(self.past_segmentations)
            time_since_segmentation_start = int(time.time() - self.data_acquisition_start_time) if self.data_acquisition_start_time else 0
            file_name = f"{self.sub_directory}/{formatted_date_time}_seg{segmentation_number}_time{time_since_segmentation_start}.pkl"
            with open(file_name,'wb') as file:
                pickle.dump([all_samples,self.distance,[relative_pitch,relative_roll,relative_yaw]],file)



    def apply_colormap(self, img_array, cmap_name='jet'):
        # img_array should be normalized to [0, 1]
        cmap = matplotlib.colormaps[cmap_name]
        colored_img = cmap(img_array)  # Returns RGBA
        # Convert to 8-bit unsigned integer
        colored_img = (colored_img[:, :, :3] * 255).astype(np.uint8)  # Drop alpha for pyqtgraph
        return colored_img


    def btn_pressed(self):
        self.save_data = not self.save_data
        if self.save_data:
            # Count the number of subdirectories in the save directory
            sub_directory_count = len([name for name in os.listdir(self.save_directory) if os.path.isdir(os.path.join(self.save_directory, name))]) + 1
            self.data_acquisition_count_display.setText(f"Sample Number: {sub_directory_count}")

            # Record the start time of the data acquisition session
            self.data_acquisition_start_time = time.time()

            print(f'starting to save data')
            sub_directory = os.path.join(self.save_directory, str(int(time.time())))
            sub_directory += '_' + self.annotation_text + str(sub_directory_count)
            os.makedirs(sub_directory, exist_ok=True)
            print(f'Created sub-directory: {sub_directory}')
            self.sub_directory = sub_directory
        else:
            print(f'not saving data')
        

    def toggle_segmentation(self):
        if not hasattr(self, 'segmentation_enabled'):
            self.segmentation_enabled = False
        self.segmentation_enabled = not self.segmentation_enabled
        print(f"Segmentation enabled: {self.segmentation_enabled}")

        if self.segmentation_enabled:
            pass
        else:
            # Remove all highlighted regions if segmentation is disabled
            if hasattr(self, 'highlighted_regions'):
                for region in self.highlighted_regions:
                    self.plot_widget.removeItem(region)
                self.highlighted_regions = []


    def on_annotation_text_changed(self, text):
        self.annotation_text = text

    def on_threshold_changed(self, value):
        self.segmentation_threshold = value
        print(f"Segmentation threshold updated to: {self.segmentation_threshold}")

    def toggle_angle_display(self):
        self.show_angle = not self.show_angle
        self.initial_angle = np.asarray(self.sensor.euler)
        self.initial_quat = R.from_quat(self.sensor.quaternion)  # Store initial quaternion
        
        # Reset the background image when toggling angle display
        self.bg_image_array = np.zeros((self.image_dimensions, self.image_dimensions), dtype=float)
        self.colored_img = np.zeros(self.colored_img.shape).astype(np.uint8)  # Reset colored image
        self.bg_image.setImage(self.bg_image_array, autoLevels=False)


        if self.show_angle:
            self.toggle_angle_btn.setText('Hide Angle')
            # Center cross-hairs
            self.crosshair_v.setPos(0)
            self.crosshair_h.setPos(0)
        else:
            self.toggle_angle_btn.setText('Show Angle')
        if not self.show_angle:
            # self.angle_line.setData([], [])
            # Optionally hide cross-hairs by moving them off-plot:
            self.crosshair_v.setPos(0)
            self.crosshair_h.setPos(0)

    def toggle_colormap(self):
        self.use_colormap = not self.use_colormap
        if self.use_colormap:
            self.colormap_toggle_btn.setText('Show Greyscale')
            self.colored_img = self.apply_colormap(self.blurred_image_array, cmap_name='jet')
            self.bg_image.setImage(self.colored_img, autoLevels=False)
        else:
            self.colormap_toggle_btn.setText('Show Colormap')
            self.bg_image.setImage(self.blurred_image_array, autoLevels=False)


    def qt_connections(self):
        self.btn.pressed.connect(self.btn_pressed)
        self.segmentation_btn.pressed.connect(self.toggle_segmentation)
        self.annotation_input.textChanged.connect(self.on_annotation_text_changed)
        self.threshold_spinbox.valueChanged.connect(self.on_threshold_changed)  # Connect the spin box
        self.toggle_angle_btn.pressed.connect(self.toggle_angle_display)  # Add this line
        self.colormap_toggle_btn.pressed.connect(self.toggle_colormap)

    def plot_coords_to_image_indices(self, x, y):
        # Map x, y in [-1, 1] to image indices [0, image_dimensions-1]
        ix = int((x + 1) / 2 * (self.image_dimensions - 1))
        iy = int((y + 1) / 2 * (self.image_dimensions - 1))
        # Clamp to valid range
        ix = np.clip(ix, 0, self.image_dimensions - 1)
        iy = np.clip(iy, 0, self.image_dimensions - 1)
        return ix, iy

    def on_model_weights_changed(self, idx):
        weights_file = self.model_weights_files[idx]
        self.segmentation_model.load_state_dict(torch.load(weights_file))
        self.segmentation_model.eval()
        print(f"Loaded model weights: {weights_file}")



    def run(self):
        sys.exit(self.app.exec())

if __name__ == '__main__':
    plotter = TimeSeriesPlotter()
    plotter.run()

