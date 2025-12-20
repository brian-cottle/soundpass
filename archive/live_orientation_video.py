import sys
import numpy as np
from PyQt6.QtWidgets import QApplication
from PyQt6 import QtWidgets
from PyQt6.QtCore import QTimer
import pyqtgraph as pg
import numpy as np
import time
from glob import glob
from datetime import datetime
import pickle
from PyQt6.QtCore import Qt
import os
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as R
from skimage.filters import gaussian
import matplotlib
import matplotlib.cm
import cv2
from pathlib import Path
from natsort import natsorted

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

class TimeSeriesPlayer:
    def __init__(self, data_directory):
        self.app = QApplication(sys.argv)
        self.data_directory = data_directory
        self.current_file_index = 0
        
        # Get sorted list of pickle files
        self.pickle_files = natsorted(glob(os.path.join(data_directory, "*.pkl")))
        if not self.pickle_files:
            raise ValueError(f"No pickle files found in {data_directory}")
        
        self.setup_gui()
        
        # Setup video writer
        self.frame_size = (1920, 1080)  # Adjust based on your window size
        self.video_writer = None
        self.output_video_path = os.path.join(data_directory, "playback.mp4")
        print(self.output_video_path)
        
        self.load_first_file()

        # Setup playback timer (10Hz)
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self.update_plot)
        self.playback_timer.start(100)  # 100ms = 10Hz

    def setup_gui(self):
        self.w = QtWidgets.QWidget()
        self.w.setWindowTitle('Data Playback')
        layout = QtWidgets.QGridLayout()
        self.w.setLayout(layout)

        # Main plot widget
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget, 0, 1, 8, 2)
        self.plot_widget.getPlotItem().getAxis('bottom').setLabel('Distance (cm)', **{'font-size': '25px'})
        axis = self.plot_widget.getPlotItem().getAxis('bottom')
        axis.setStyle(tickFont=pg.QtGui.QFont('Arial', 45))
        axis.setPen(pg.mkPen(color='w'))

        # Polar plot
        self.polar_plot = pg.PlotWidget()
        self.polar_plot.setAspectLocked(True)
        self.polar_plot.hideAxis('left')
        self.polar_plot.hideAxis('bottom')
        self.polar_plot.setBackground('k')
        layout.addWidget(self.polar_plot, 0, 3, 8, 1)

        # Setup background image for polar plot
        self.image_dimensions = 100
        self.bg_image_array = np.zeros((self.image_dimensions, self.image_dimensions), dtype=float)
        self.blurred_image_array = gaussian(self.bg_image_array, sigma=2)
        self.bg_image = pg.ImageItem(self.bg_image_array)
        self.bg_image.setRect(pg.QtCore.QRectF(-1, -1, 2, 2))
        self.polar_plot.addItem(self.bg_image)

        # Add crosshairs
        self.setup_crosshairs()

        # Main plot
        self.plot1 = self.plot_widget.plot(pen='y')
        self.plot1.setAlpha(0.4, False)
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.plot_widget.setYRange(-10000, 10000)

        # Show the window
        self.w.show()

    def setup_crosshairs(self):
        # Moving crosshairs
        self.crosshair_v = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('y', width=2))
        self.crosshair_h = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('y', width=2))
        self.polar_plot.addItem(self.crosshair_v)
        self.polar_plot.addItem(self.crosshair_h)

        # Static crosshairs
        static_pen = pg.mkPen(color=(150, 150, 150, 120), width=2, style=Qt.PenStyle.DotLine)
        self.static_crosshair_v = pg.InfiniteLine(angle=90, movable=False, pen=static_pen)
        self.static_crosshair_h = pg.InfiniteLine(angle=0, movable=False, pen=static_pen)
        self.polar_plot.addItem(self.static_crosshair_v)
        self.polar_plot.addItem(self.static_crosshair_h)
        self.static_crosshair_v.setPos(0)
        self.static_crosshair_h.setPos(0)

    def load_first_file(self):
        if self.pickle_files:
            with open(self.pickle_files[0], 'rb') as f:
                data = pickle.load(f)
                self.distance = data[1]  # Extract distance array
                self.setup_video_writer()

    def setup_video_writer(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            self.output_video_path,
            fourcc,
            10.0,  # 10 FPS
            self.frame_size
        )

    def capture_frame(self):
        # Capture the current window content
        screen = self.w.grab()
        image = screen.toImage()
        width = image.width()
        height = image.height()
        ptr = image.bits()
        ptr.setsize(height * width * 4)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
        frame = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        
        # Resize to match video dimensions
        frame = cv2.resize(frame, self.frame_size)
        self.video_writer.write(frame)

    def plot_coords_to_image_indices(self, x, y):
        ix = int((x + 1) / 2 * (self.image_dimensions - 1))
        iy = int((y + 1) / 2 * (self.image_dimensions - 1))
        ix = np.clip(ix, 0, self.image_dimensions - 1)
        iy = np.clip(iy, 0, self.image_dimensions - 1)
        return ix, iy

    def update_plot(self):
        if self.current_file_index >= len(self.pickle_files):
            print("Playback complete")
            self.playback_timer.stop()
            if self.video_writer:
                self.video_writer.release()
            self.app.quit()
            return

        # Load current file
        with open(self.pickle_files[self.current_file_index], 'rb') as f:
            data = pickle.load(f)

        # Unpack data
        all_samples = data[0]
        distance = data[1]
        orientation = data[2]  # [relative_pitch, relative_roll, relative_yaw]

        # Update signal plot
        average_samples = np.mean(all_samples, axis=0)
        self.plot1.setData(self.distance, average_samples)

        # Update orientation visualization
        relative_pitch = orientation[0]
        relative_roll = orientation[1]
        relative_yaw = orientation[2]

        # Calculate tip position
        d = 2
        pitch_rad = np.radians(relative_pitch)
        yaw_rad = np.radians(relative_yaw)
        y = d * np.sin(yaw_rad) * np.cos(pitch_rad)
        x = d * np.sin(pitch_rad)

        # Update crosshairs
        self.crosshair_v.setPos(-x)
        self.crosshair_h.setPos(-y)

        # Update orientation trace
        ix, iy = self.plot_coords_to_image_indices(-y, -x)
        self.bg_image_array[iy, ix] = 1.0
        self.blurred_image_array = gaussian(self.bg_image_array, sigma=2)
        self.bg_image.setImage(self.blurred_image_array, autoLevels=True)

        # Capture frame for video
        self.capture_frame()

        # Move to next file
        self.current_file_index += 1

    def run(self):
        sys.exit(self.app.exec())

if __name__ == '__main__':
    data_dir = "/home/briancottle/Code/SoundPass/new_datasets/2025-05-30/1748623248_12  "  # Replace with your data directory
    player = TimeSeriesPlayer(data_dir)
    player.run()