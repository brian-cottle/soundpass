import os
import pickle
import numpy as np
from natsort import natsorted
from datetime import datetime
from glob import glob

from PyQt6 import QtWidgets, QtCore
from PyQt6.QtCore import Qt
import pyqtgraph as pg
from PyQt6.QtWidgets import QPushButton, QVBoxLayout
from PyQt6.QtCore import Qt
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d

import torch
import torch.nn as nn
import pywt

def low_pass_filter(data, cutoff=10, fs=1000, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# Define a simple 1D segmentation network
class SegmentationNetwork(nn.Module):
    def __init__(self, input_channels, num_classes, kernel_size=45, hidden_channels=85):
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


class AnnotateSignalSlider(QtWidgets.QWidget):
    def __init__(self, top_level_directory, title="Signal Annotator with Slider"):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(1200, 800)
        self.top_level_directory = top_level_directory

        layout = QtWidgets.QVBoxLayout(self)

        # Find all immediate subdirectories
        self.directories = [
            os.path.join(self.top_level_directory, d)
            for d in os.listdir(self.top_level_directory)
            if os.path.isdir(os.path.join(self.top_level_directory, d))
        ]
        self.directories = natsorted(self.directories)

        # Segmentation setup
        self.segmentation_enabled = False
        self.highlighted_regions = []
        self.segmentation_threshold = 150
        self.past_segmentations = []
        self.front_plot_threshold = 7000


        # Directory selector
        self.dir_selector = QtWidgets.QComboBox()
        self.dir_selector.addItems(self.directories)
        self.dir_selector.currentIndexChanged.connect(self.directory_changed)
        layout.addWidget(self.dir_selector)

        # File name display
        self.file_label = QtWidgets.QLabel("Current file: ")
        self.file_label.setStyleSheet("font-size: 18pt; color: white;")
        layout.addWidget(self.file_label)

        # Plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('k')
        self.plot_widget.getPlotItem().getAxis('bottom').setLabel('Distance', **{'font-size': '18pt'})
        layout.addWidget(self.plot_widget)

        # Slider
        self.slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.valueChanged.connect(self.slider_changed)
        layout.addWidget(self.slider)

        # Toggle segmentation button
        self.segmentation_btn = QPushButton("Toggle Segmentation")
        self.segmentation_btn.clicked.connect(self.toggle_segmentation)
        layout.addWidget(self.segmentation_btn)

        # Segmentation threshold input (QDoubleSpinBox with step 0.1)
        self.threshold_input = QtWidgets.QDoubleSpinBox()
        self.threshold_input.setDecimals(0)
        self.threshold_input.setSingleStep(50)
        self.threshold_input.setRange(0, 1000)
        self.threshold_input.setValue(self.segmentation_threshold)
        self.threshold_input.setFixedWidth(120)
        self.threshold_input.setStyleSheet("font-size: 16pt;")
        self.threshold_input.valueChanged.connect(self.update_threshold)
        layout.addWidget(self.threshold_input)

        # Initialize file loading
        self.current_index = 0
        if self.directories:
            self.load_files_from_directory(self.directories[0])
        else:
            self.file_names = []
            self.all_data_array = []
            self.all_file_names = []
            self.distances = []

        # For annotation
        self.starts = []
        self.ends = []
        self.regions = []
        self.current_file_name = None

        # Save directory
        current_date = datetime.now().strftime("%Y-%m-%d")
        self.save_dir = f"./{current_date}_ground_truth"
        os.makedirs(self.save_dir, exist_ok=True)

        # Initialize weighting array
        weighting_array = np.zeros(len(self.distance))
        weighting_array[2500:17500] = 1
        # Apply Gaussian filter to smooth the weighting array
        sigma = 1000  # Standard deviation for the Gaussian filter
        weighting_array = gaussian_filter1d(weighting_array, sigma=sigma)
        self.weighting_array = weighting_array

        self.plot_data()




    def directory_changed(self, idx):
        directory = self.directories[idx]
        self.load_files_from_directory(directory)
        self.current_index = 0
        self.slider.setValue(0)
        self.plot_data()

    def load_files_from_directory(self, directory):
        self.file_names = natsorted(glob(os.path.join(directory, "*.pkl")))
        self.all_data_array = []
        self.all_file_names = []
        self.distances = []
        for file_name in self.file_names:
            with open(file_name, 'rb') as f:
                data = pickle.load(f)
                averaged_data = np.mean(data[0], axis=0)
                self.all_data_array.append(averaged_data)
                self.distances.append(data[1])
                self.all_file_names.append(file_name)
        self.slider.setMaximum(max(0, len(self.file_names) - 1))
        self.distance = self.distances[0][self.front_plot_threshold:]

    def slider_changed(self, value):
        self.current_index = value
        self.plot_data()

    def plot_data(self):
        self.plot_widget.clear()
        self.starts = []
        self.ends = []
        self.regions = []
        if not self.all_data_array:
            self.file_label.setText("No files in directory.")
            return
        self.data = self.all_data_array[self.current_index][self.front_plot_threshold:]  
        self.original_data = self.all_data_array[self.current_index][self.front_plot_threshold:]
        self.distance = self.distances[self.current_index][self.front_plot_threshold:]
        # Apply a low-pass filter to the data
        filter_data = low_pass_filter(self.data, cutoff=20, fs=1000, order=5)

        abs_data = np.abs(filter_data)
        wavelet = 'db4'
        coeffs = pywt.wavedec(filter_data, wavelet, level=4)
        # Apply thresholding to detail coefficients (skip the first, which is the approximation)
        threshold_value = 0.04 * np.max([np.max(np.abs(c)) for c in coeffs[1:]])  # Example threshold, can be tuned
        thresholded_coeffs = [coeffs[0]] + [pywt.threshold(c, threshold_value, mode='soft') for c in coeffs[1:]]

        wavelet_data = pywt.waverec(thresholded_coeffs, wavelet)
        self.extracted_wavelet_data = np.abs(abs_data - wavelet_data)
        self.weighted_data = self.extracted_wavelet_data * self.weighting_array

        self.current_file_name = self.all_file_names[self.current_index]
        # Plot unfiltered data
        self.plot_widget.plot(self.distance, self.original_data, pen=pg.mkPen(color=(0, 255, 0, 50)))
        self.plot_widget.plot(self.distance, self.weighted_data, pen=pg.mkPen(color=(255, 0, 0, 250)))
        # Set y-axis range
        self.plot_widget.setYRange(0, 400)  # You can adjust these values as needed
        self.file_label.setText(f"Current file: {self.current_index + 1}/{len(self.all_file_names)} - {os.path.basename(self.current_file_name)}")

        if self.segmentation_enabled:
            self.perform_segmentation()

    def perform_segmentation(self):
        binary_prediction = (self.weighted_data > self.segmentation_threshold).astype(int)
        # Perform binary closing to fill gaps between segments
        kernel_size = 500  # Adjust this value to control the size of gaps to fill
        kernel = np.ones(kernel_size)
        binary_prediction = np.convolve(binary_prediction, kernel, mode='same')
        binary_prediction = (binary_prediction > 0).astype(int)  # Convert back to binary
        
        
        # Update the region based on the predicted class
        if hasattr(self, 'highlighted_regions'):
            for region in self.highlighted_regions:
                self.plot_widget.removeItem(region)
        self.highlighted_regions = []

        segment_indices = np.where(binary_prediction == 1)[0]
        if len(segment_indices) > 0:
            # Find contiguous regions of positive values
            regions = np.split(segment_indices, np.where(np.diff(segment_indices) != 1)[0] + 1)
            for region in regions:
                start_index = region[0]
                end_index = region[-1]
                start_pos = self.distance[start_index]
                end_pos = self.distance[end_index]
                # Create a new region for each contiguous segment
                new_region = pg.LinearRegionItem([start_pos, end_pos])
                new_region.setZValue(10)  # Ensure it appears above the plot
                new_region.setBrush(pg.mkBrush(255, 0, 0, 50))  # Semi-transparent red
                self.plot_widget.addItem(new_region)
                self.highlighted_regions.append(new_region)

    def toggle_segmentation(self):
        self.segmentation_enabled = not self.segmentation_enabled
        print(f"Segmentation enabled: {self.segmentation_enabled}")
        self.plot_data()

    def update_threshold(self):
        self.segmentation_threshold = self.threshold_input.value()
        print(f"Segmentation threshold updated to: {self.segmentation_threshold}")
        if self.segmentation_enabled:
            self.plot_data()


if __name__ == "__main__":
    import sys

    # Example usage: pass a single top-level directory
    top_dir = "/home/briancottle/Code/SoundPass/new_datasets/2025-05-30/"
    app = QtWidgets.QApplication(sys.argv)
    window = AnnotateSignalSlider(top_dir)
    window.show()
    sys.exit(app.exec())