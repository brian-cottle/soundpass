import sys
import numpy as np
from PyQt6.QtWidgets import QApplication
from PyQt6 import QtWidgets
from PyQt6.QtCore import QTimer, Qt
import pyqtgraph as pg
import time
from glob import glob
from datetime import datetime
import pickle
import os
from scipy.spatial.transform import Rotation as R
from skimage.filters import gaussian
import matplotlib
import matplotlib.cm
import cv2
from pathlib import Path
from natsort import natsorted
import pywt
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d


def low_pass_filter(data, cutoff=10, fs=1000, order=5):
    """Apply low-pass filter to data"""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def high_pass_filter(data, cutoff=10, fs=1000, order=5):
    """Apply high-pass filter to data"""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)

def downsample_data(x_data, y_data, factor):
    """Downsample data by the given factor"""
    if factor <= 1:
        return x_data, y_data
    
    # Simple downsampling by taking every nth point
    downsampled_x = x_data[::factor]
    downsampled_y = y_data[::factor]
    
    return downsampled_x, downsampled_y




class BaseVisualizationMixin:
    """Mixin class containing common visualization methods"""
    
    def initialize_common_variables(self):
        """Initialize common variables used by both live and playback versions"""
        self.segmentation_enabled = False
        self.segmentation_threshold = 150
        self.show_angle = False
        self.use_colormap = True
        self.save_data = False
        self.annotation_text = ''
        self.data_acquisition_start_time = None
        self.highlighted_regions = []
        self.current_segmentation_active = False
        self.current_segmentation_intensity = 0.0
        self.refresh_rate_array = [0]*20
        
        # Reference markers for polar plot
        self.reference_markers = []
        self.reference_marker_items = []
        
        # Initialize time tracking
        self.t_start = time.time()
        
        # Wavelet processing parameters
        self.front_plot_threshold = 800
        
        # Initialize weighting array
        self.weighting_array = None
        
        # Track maximum values to preserve high intensity regions
        self.max_intensity_array = None
        
        # Performance optimization variables
        self.polar_update_counter = 0
        self.polar_update_frequency = 1  # Update polar plot every 3 frames
        self.blur_counter = 0
        self.blur_frequency = 5  # Apply blur every 5 polar updates
        
        # Y-axis range parameters
        self.y_axis_min = -4000
        self.y_axis_max = 4000
        
        # Plot downsampling parameters
        self.enable_plot_downsampling = False
        self.plot_downsample_factor = 1
        
        # Window sizing parameters for different screen resolutions
        self.window_width = 1400  # Default window width
        self.window_height = 900  # Default window height
        self.button_width = 300   # Default button width
        self.button_height = 150  # Default button height
        self.control_width = 300  # Default control width
        self.control_height = 80  # Default control height
        self.font_size = 20       # Default font size

    def configure_window_sizing(self, screen_resolution="medium"):
        """Configure window sizing based on screen resolution
        
        Args:
            screen_resolution (str): "small", "medium", or "large"
        """
        if screen_resolution == "small":
            # For small screens (e.g., 1024x768, 1366x768)
            self.window_width = 1000
            self.window_height = 700
            self.button_width = 200
            self.button_height = 100
            self.control_width = 200
            self.control_height = 60
            self.font_size = 16
        elif screen_resolution == "medium":
            # For medium screens (e.g., 1440x900, 1600x900)
            self.window_width = 1400
            self.window_height = 900
            self.button_width = 300
            self.button_height = 150
            self.control_width = 300
            self.control_height = 80
            self.font_size = 20
        elif screen_resolution == "large":
            # For large screens (e.g., 1920x1080, 2560x1440)
            self.window_width = 1800
            self.window_height = 1200
            self.button_width = 400
            self.button_height = 200
            self.control_width = 400
            self.control_height = 100
            self.font_size = 25

    def setup_window_sizing(self, window_widget):
        """Setup window sizing and properties"""
        window_widget.resize(self.window_width, self.window_height)
        window_widget.setMinimumSize(800, 600)  # Set minimum size to prevent too small windows
        # Center the window on screen
        screen = QApplication.primaryScreen().geometry()
        x = (screen.width() - self.window_width) // 2
        y = (screen.height() - self.window_height) // 2
        window_widget.move(x, y)

    def initialize_weighting_array(self, distance):
        """Initialize weighting array for wavelet processing"""
        if len(distance) > self.front_plot_threshold:
            distance_cropped = distance[self.front_plot_threshold:]
            weighting_array = np.zeros(len(distance_cropped))
            # Set weights for middle section (adjust indices as needed)
            start_idx = 500 # default is 2500
            end_idx = 7250 # default is 17500
            if start_idx < end_idx:
                weighting_array[start_idx:end_idx] = 1
            # Apply Gaussian filter to smooth the weighting array
            sigma = 65 # default is 1000
            self.weighting_array = gaussian_filter1d(weighting_array, sigma=sigma)
        else:
            self.weighting_array = np.ones(len(distance))

    def setup_left_controls(self, layout):
        """Setup left panel controls"""
        row = 0
        
        # Data Acquisition toggle
        self.btn = QtWidgets.QPushButton('Save Data')
        self.btn.setFixedSize(self.button_width, self.button_height)
        self.btn.setStyleSheet(f"background-color: #F44336; color: white; font-size: {self.font_size}px; border-radius: 10px; padding: 10px;")
        layout.addWidget(self.btn, row, 0)
        row += 1
        
        # Segmentation toggle
        self.segmentation_btn = QtWidgets.QPushButton('Segmentation')
        self.segmentation_btn.setFixedSize(self.button_width, self.button_height)
        self.segmentation_btn.setStyleSheet(f"background-color: #F44336; color: white; font-size: {self.font_size}px; border-radius: 10px; padding: 10px;")
        layout.addWidget(self.segmentation_btn, row, 0)
        row += 1
        
        # Angle display toggle
        self.toggle_angle_btn = QtWidgets.QPushButton('Orientation')
        self.toggle_angle_btn.setFixedSize(self.button_width, self.button_height)
        self.toggle_angle_btn.setStyleSheet(f"background-color: #F44336; color: white; font-size: {self.font_size}px; border-radius: 10px; padding: 10px;")
        layout.addWidget(self.toggle_angle_btn, row, 0)
        row += 1
        
        # Clear markers button
        self.clear_markers_btn = QtWidgets.QPushButton('Clear Reference Markers')
        self.clear_markers_btn.setFixedSize(self.button_width, self.control_height)
        self.clear_markers_btn.setStyleSheet(f"background-color: #607D8B; color: white; font-size: {self.font_size}px; border-radius: 10px; padding: 10px;")
        layout.addWidget(self.clear_markers_btn, row, 0)
        row += 1
        
        # Annotation input
        self.annotation_input = QtWidgets.QLineEdit()
        self.annotation_input.setPlaceholderText("Enter annotation here...")
        self.annotation_input.setFixedSize(self.control_width, self.control_height)
        self.annotation_input.setStyleSheet(f"font-size: {self.font_size + 5}px; padding: 10px;")
        layout.addWidget(self.annotation_input, row, 0)
        row += 1
        
        # Refresh rate display
        self.refresh_rate_display = QtWidgets.QLineEdit()
        self.refresh_rate_display.setReadOnly(True)
        self.refresh_rate_display.setFixedSize(self.control_width, self.control_height)
        self.refresh_rate_display.setStyleSheet(f"font-size: {self.font_size + 5}px; padding: 10px;")
        layout.addWidget(self.refresh_rate_display, row, 0)
        row += 1
        
        # Data acquisition count display
        self.data_acquisition_count_display = QtWidgets.QLineEdit()
        self.data_acquisition_count_display.setReadOnly(True)
        self.data_acquisition_count_display.setFixedSize(self.control_width, self.control_height)
        self.data_acquisition_count_display.setStyleSheet(f"font-size: {self.font_size + 5}px; padding: 10px;")
        self.data_acquisition_count_display.setText("Sample Number: 0")
        layout.addWidget(self.data_acquisition_count_display, row, 0)
        row += 1
        
        # Elapsed time display
        self.elapsed_time_display = QtWidgets.QLineEdit()
        self.elapsed_time_display.setReadOnly(True)
        self.elapsed_time_display.setFixedSize(self.control_width, self.control_height)
        self.elapsed_time_display.setStyleSheet(f"font-size: {self.font_size + 5}px; padding: 10px;")
        self.elapsed_time_display.setText("Elapsed Time: 0m 0s")
        layout.addWidget(self.elapsed_time_display, row, 0)
        row += 1
        
        # Segmentation threshold spinbox
        self.threshold_spinbox = QtWidgets.QSpinBox()
        self.threshold_spinbox.setSingleStep(50)
        self.threshold_spinbox.setRange(0, 1000)
        self.threshold_spinbox.setValue(150)
        self.threshold_spinbox.setFixedSize(self.control_width, self.control_height)
        self.threshold_spinbox.setStyleSheet(
            f"font-size: {self.font_size + 5}px; padding: 10px; "
            "QAbstractSpinBox::up-arrow { width: 50px; height: 50px; } "
            "QAbstractSpinBox::down-arrow { width: 50px; height: 50px; }"
        )
        layout.addWidget(self.threshold_spinbox, row, 0)
        row += 1
        
        # Front plot threshold label
        self.front_threshold_label = QtWidgets.QLabel('Front Plot Threshold:')
        self.front_threshold_label.setFixedSize(self.control_width, 40)
        self.front_threshold_label.setStyleSheet(f"font-size: {self.font_size - 5}px; color: black; padding: 5px;")
        layout.addWidget(self.front_threshold_label, row, 0)
        row += 1
        
        # Front plot threshold spinbox
        self.front_threshold_spinbox = QtWidgets.QSpinBox()
        self.front_threshold_spinbox.setSingleStep(100)
        self.front_threshold_spinbox.setRange(0, 10000)
        self.front_threshold_spinbox.setValue(self.front_plot_threshold)
        self.front_threshold_spinbox.setFixedSize(self.control_width, self.control_height)
        self.front_threshold_spinbox.setStyleSheet(
            f"font-size: {self.font_size + 5}px; padding: 10px; "
            "QAbstractSpinBox::up-arrow { width: 50px; height: 50px; } "
            "QAbstractSpinBox::down-arrow { width: 50px; height: 50px; }"
        )
        layout.addWidget(self.front_threshold_spinbox, row, 0)
        row += 1
        
        # Y-axis min label
        self.y_axis_min_label = QtWidgets.QLabel('Y-axis Minimum:')
        self.y_axis_min_label.setFixedSize(self.control_width, 40)
        self.y_axis_min_label.setStyleSheet(f"font-size: {self.font_size - 5}px; color: black; padding: 5px;")
        layout.addWidget(self.y_axis_min_label, row, 0)
        row += 1
        
        # Y-axis min input
        self.y_axis_min_input = QtWidgets.QLineEdit()
        self.y_axis_min_input.setPlaceholderText("Y-axis Min (e.g., -4000)")
        self.y_axis_min_input.setText(str(self.y_axis_min))
        self.y_axis_min_input.setFixedSize(self.control_width, self.control_height)
        self.y_axis_min_input.setStyleSheet(f"font-size: {self.font_size + 5}px; padding: 10px;")
        layout.addWidget(self.y_axis_min_input, row, 0)
        row += 1
        
        # Y-axis max label
        self.y_axis_max_label = QtWidgets.QLabel('Y-axis Maximum:')
        self.y_axis_max_label.setFixedSize(self.control_width, 40)
        self.y_axis_max_label.setStyleSheet(f"font-size: {self.font_size - 5}px; color: black; padding: 5px;")
        layout.addWidget(self.y_axis_max_label, row, 0)
        row += 1
        
        # Y-axis max input
        self.y_axis_max_input = QtWidgets.QLineEdit()
        self.y_axis_max_input.setPlaceholderText("Y-axis Max (e.g., 4000)")
        self.y_axis_max_input.setText(str(self.y_axis_max))
        self.y_axis_max_input.setFixedSize(self.control_width, self.control_height)
        self.y_axis_max_input.setStyleSheet(f"font-size: {self.font_size + 5}px; padding: 10px;")
        layout.addWidget(self.y_axis_max_input, row, 0)

    def setup_main_plot(self, layout):
        """Setup main plot widgets - original signal and wavelet processed"""
        # Original signal plot (top half)
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget, 0, 2, 8, 1)
        
        # Add title to main plot
        self.plot_widget.setTitle("Original Signal", color='w', size=f'{self.font_size}px')
        
        self.plot_widget.getPlotItem().getAxis('bottom').setLabel('Distance (cm)', **{'font-size': f'{self.font_size + 5}px'})
        axis = self.plot_widget.getPlotItem().getAxis('bottom')
        axis.setStyle(tickFont=pg.QtGui.QFont('Arial', self.font_size + 20))
        axis.setPen(pg.mkPen(color='w'))
        
        self.plot1 = self.plot_widget.plot(pen='y')
        self.plot1.setAlpha(0.4, False)
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.plot_widget.setYRange(self.y_axis_min, self.y_axis_max)
        
        # Wavelet processed signal plot (bottom half)
        self.wavelet_plot_widget = pg.PlotWidget()
        layout.addWidget(self.wavelet_plot_widget, 8, 2, 8, 1)
        
        # Add title to wavelet plot
        self.wavelet_plot_widget.setTitle("Processed Signal", color='w', size=f'{self.font_size}px')
        
        self.wavelet_plot_widget.getPlotItem().getAxis('bottom').setLabel('Distance (cm)', **{'font-size': f'{self.font_size + 5}px'})
        wavelet_axis = self.wavelet_plot_widget.getPlotItem().getAxis('bottom')
        wavelet_axis.setStyle(tickFont=pg.QtGui.QFont('Arial', self.font_size + 20))
        wavelet_axis.setPen(pg.mkPen(color='w'))
        
        self.wavelet_plot1 = self.wavelet_plot_widget.plot(pen='r')
        self.wavelet_plot_widget.setMouseEnabled(x=False, y=False)
        self.wavelet_plot_widget.setYRange(0, 350)

    def setup_polar_plot(self, layout):
        """Setup polar plot for orientation visualization"""
        self.polar_plot = pg.PlotWidget()
        self.polar_plot.setAspectLocked(True)
        self.polar_plot.hideAxis('left')
        self.polar_plot.hideAxis('bottom')
        self.polar_plot.setBackground('k')
        layout.addWidget(self.polar_plot, 0, 4, 16, 1)
        
        # Setup background image
        self.image_dimensions = 100
        self.bg_image_array = np.zeros((self.image_dimensions, self.image_dimensions), dtype=float)
        self.bg_image = pg.ImageItem(self.bg_image_array)
        self.bg_image.setRect(pg.QtCore.QRectF(-1, -1, 2, 2))
        self.polar_plot.addItem(self.bg_image)
        
        # Initialize colored image
        self.colored_img = np.zeros((self.image_dimensions, self.image_dimensions, 3), dtype=np.uint8)
        
        # Initialize max intensity tracking array
        self.max_intensity_array = np.zeros((self.image_dimensions, self.image_dimensions), dtype=float)
        
        # Add polar grid
        self.polar_grid = pg.ScatterPlotItem(
            x=[np.cos(np.radians(angle)) for angle in range(0, 360, 30)],
            y=[np.sin(np.radians(angle)) for angle in range(0, 360, 30)],
            pen=pg.mkPen('w'),
            brush=pg.mkBrush(255, 255, 255, 50),
            size=5
        )
        self.polar_plot.addItem(self.polar_grid)
        
        # Add crosshairs
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
        
        # Connect mouse click event for reference markers
        self.polar_plot.scene().sigMouseClicked.connect(self.on_polar_plot_clicked)

    def setup_common_connections(self):
        """Setup common signal connections"""
        self.segmentation_btn.pressed.connect(self.toggle_segmentation)
        self.toggle_angle_btn.pressed.connect(self.toggle_angle_display)
        self.clear_markers_btn.pressed.connect(self.clear_reference_markers)
        self.annotation_input.textChanged.connect(self.on_annotation_text_changed)
        self.threshold_spinbox.valueChanged.connect(self.on_threshold_changed)
        self.front_threshold_spinbox.valueChanged.connect(self.on_front_threshold_changed)
        self.y_axis_min_input.textChanged.connect(self.on_y_axis_min_changed)
        self.y_axis_max_input.textChanged.connect(self.on_y_axis_max_changed)

    def main_update(self):
        """Main update loop for GUI"""
        # Update refresh rate display
        self.refresh_rate_array.append(1 / (time.time() - self.t_start))
        self.refresh_rate_array.pop(0)
        refresh_rate = np.mean(self.refresh_rate_array)
        self.refresh_rate_display.setText(f'Refresh rate (Hz): {refresh_rate:0.3f}')
        self.t_start = time.time()

    def process_wavelet_data(self, signal_data):
        """Process signal data using wavelet transform"""
        try:
            # Crop the data based on front_plot_threshold
            if hasattr(self, 'distance') and self.distance is not None:
                if len(signal_data) > self.front_plot_threshold:
                    cropped_data = signal_data[self.front_plot_threshold:]
                    cropped_distance = self.distance[self.front_plot_threshold:]
                else:
                    cropped_data = signal_data
                    cropped_distance = self.distance
            else:
                cropped_data = signal_data
                cropped_distance = None

            # Use the same wavelet and level as in the proper wavelet sampling method
            wavelet = 'db4'
            level = 5

            # Perform MRA (multiresolution analysis) to get the detail signals at each level
            mra_coeffs = pywt.mra(cropped_data, wavelet, level=level, transform='dwt')
            mra_coeffs = np.array(mra_coeffs)
            mra_coeffs[0:2,:] = 0
            
            # Take the absolute value of the signal
            abs_mra_coeffs = np.abs(mra_coeffs)

            original_signal = np.sum(abs_mra_coeffs, axis=0)
            low_passed_original_signal = low_pass_filter(original_signal, cutoff=15, fs=1000, order=5)

            extracted_wavelet_data = low_passed_original_signal

            # Apply weighting if available
            if self.weighting_array is not None and len(extracted_wavelet_data) == len(self.weighting_array):
                weighted_data = extracted_wavelet_data * self.weighting_array
            else:
                weighted_data = extracted_wavelet_data
            
            weighted_data[-250:] = 0

            return weighted_data, cropped_distance
            
        except Exception as e:
            print(f"Error in wavelet processing: {e}")
            return None, None

    def process_segmentation(self, wavelet_data, wavelet_distance):
        """Process segmentation using threshold-based approach"""
        try:
            if wavelet_data is None or wavelet_distance is None:
                self.current_segmentation_active = False
                self.current_segmentation_intensity = 0.0
                return
            
            # Apply threshold to wavelet data
            binary_prediction = (wavelet_data > self.segmentation_threshold).astype(int)
            
            # Perform binary closing to fill gaps between segments
            kernel_size = 500
            kernel = np.ones(kernel_size)
            binary_prediction = np.convolve(binary_prediction, kernel, mode='same')
            binary_prediction = (binary_prediction > 0).astype(int)
            
            # Update segmentation state
            self.current_segmentation_active = np.any(binary_prediction > 0)
            
            # Calculate area under curve where segmentation is positive
            if self.current_segmentation_active:
                positive_indices = binary_prediction > 0
                area_under_curve = np.sum(wavelet_data[positive_indices])
                
                max_value = np.max(wavelet_data) if len(wavelet_data) > 0 else 1.0
                num_positive = np.sum(positive_indices)
                
                if max_value > 0 and num_positive > 0:
                    self.current_segmentation_intensity = max_value * area_under_curve / len(wavelet_data)
                else:
                    self.current_segmentation_intensity = 0.0
            else:
                self.current_segmentation_intensity = 0.0
            
            self.current_segmentation_intensity = min(1.0, self.current_segmentation_intensity/750)
            # print(self.current_segmentation_intensity)
            # Update highlighted regions
            self.update_highlighted_regions_wavelet(binary_prediction, wavelet_distance)
            
        except Exception as e:
            print(f"Error in segmentation: {e}")
            self.current_segmentation_active = False
            self.current_segmentation_intensity = 0.0

    def update_highlighted_regions_wavelet(self, binary_prediction, wavelet_distance):
        """Update highlighted regions based on threshold segmentation"""
        # Remove existing regions
        for region in self.highlighted_regions:
            self.plot_widget.removeItem(region)
        self.highlighted_regions = []
        
        # Add new regions
        if binary_prediction.any() and wavelet_distance is not None:
            segment_indices = np.where(binary_prediction == 1)[0]
            if len(segment_indices) > 0:
                # Find contiguous regions of positive values
                regions = np.split(segment_indices, np.where(np.diff(segment_indices) != 1)[0] + 1)
                for region in regions:
                    start_index = region[0]
                    end_index = region[-1]
                    start_pos = wavelet_distance[start_index]
                    end_pos = wavelet_distance[end_index]
                    
                    # Create a new region for each contiguous segment
                    new_region = pg.LinearRegionItem([start_pos, end_pos])
                    new_region.setZValue(10)
                    new_region.setBrush(pg.mkBrush(255, 0, 0, 50))
                    self.plot_widget.addItem(new_region)
                    self.highlighted_regions.append(new_region)

    def calculate_tip_position(self, relative_pitch, relative_yaw):
        """Calculate tip position from pitch and yaw angles"""
        d = 2
        pitch_rad = np.radians(relative_pitch)
        yaw_rad = np.radians(relative_yaw)
        
        y = d * np.sin(yaw_rad) * np.cos(pitch_rad)
        x = d * np.sin(pitch_rad)
        
        return x, y

    def update_crosshairs(self, x, y):
        """Update crosshair positions"""
        self.crosshair_v.setPos(-x)
        self.crosshair_h.setPos(-y)

    def plot_coords_to_image_indices(self, x, y):
        """Convert plot coordinates to image indices"""
        ix = int((x + 1) / 2 * (self.image_dimensions - 1))
        iy = int((y + 1) / 2 * (self.image_dimensions - 1))
        ix = np.clip(ix, 0, self.image_dimensions - 1)
        iy = np.clip(iy, 0, self.image_dimensions - 1)
        return ix, iy

    def update_image_region(self, center_x, center_y, intensity):
        """Update a circular region around the center point with optimized performance"""
        # Only update if intensity is above threshold
        if intensity > 0.10:
            # Define circular radius for the update
            radius = 2.5
            
            # Use vectorized operations instead of nested loops
            # Create coordinate meshgrids for the circular region
            y_coords, x_coords = np.meshgrid(np.arange(-4, 5), np.arange(-4, 5), indexing='ij')
            distances = np.sqrt(x_coords**2 + y_coords**2)
            
            # Create mask for circular region
            circular_mask = distances <= radius
            
            # Get the coordinates where the mask is True
            y_offsets = y_coords[circular_mask]
            x_offsets = x_coords[circular_mask]
            
            # Calculate target coordinates
            target_x_coords = center_x + x_offsets
            target_y_coords = center_y + y_offsets
            
            # Filter for valid bounds
            valid_mask = ((target_x_coords >= 0) & (target_x_coords < self.image_dimensions) & 
                         (target_y_coords >= 0) & (target_y_coords < self.image_dimensions))
            
            if np.any(valid_mask):
                valid_x = target_x_coords[valid_mask].astype(int)
                valid_y = target_y_coords[valid_mask].astype(int)
                valid_distances = distances[circular_mask][valid_mask]
                
                # Calculate falloff intensity using vectorized operations
                falloff_intensities = intensity * np.exp(-(valid_distances**2) / (2 * 0.5**2))
                
                # Update the main image array
                self.bg_image_array[valid_y, valid_x] += falloff_intensities * 0.3
                
                # Update max intensity tracking for the affected pixels
                self.max_intensity_array[valid_y, valid_x] = np.maximum(
                    self.max_intensity_array[valid_y, valid_x], 
                    self.bg_image_array[valid_y, valid_x]
                )
            
            # Apply Gaussian blur less frequently to improve performance
            self.blur_counter += 1
            if self.blur_counter >= self.blur_frequency:
                self.blur_counter = 0
                # Apply global Gaussian blur for aesthetics
                blurred_image = gaussian(self.bg_image_array, sigma=1.0)
                
                # Preserve high intensity regions by taking maximum with the max intensity array
                self.bg_image_array = np.maximum(blurred_image, self.max_intensity_array * 0.8)
            
            # Ensure values don't exceed 1.0
            self.bg_image_array = np.clip(self.bg_image_array, 0, 1.0)

    def apply_colormap(self, img_array, cmap_name='jet'):
        """Apply colormap to image array with caching for better performance"""
        # Cache the colormap object to avoid lookup overhead
        if not hasattr(self, '_cached_cmap') or self._cached_cmap_name != cmap_name:
            self._cached_cmap = matplotlib.colormaps[cmap_name]
            self._cached_cmap_name = cmap_name
        
        colored_img = self._cached_cmap(img_array)
        # Use in-place operations where possible for better performance
        colored_img = colored_img[:, :, :3]  # Remove alpha channel
        colored_img *= 255
        return colored_img.astype(np.uint8)

    def update_polar_image(self):
        """Update polar plot image with colormap or grayscale"""
        if self.use_colormap:
            # Only apply colormap if the image has actually changed significantly
            if not hasattr(self, '_last_image_hash'):
                self._last_image_hash = 0
            
            # Use a simple hash to detect changes
            current_hash = hash(self.bg_image_array.tobytes())
            
            # Always update if image is all zeros (after reset) or if hash changed
            image_is_empty = np.allclose(self.bg_image_array, 0.0)
            
            if current_hash != self._last_image_hash or image_is_empty:
                self.colored_img = self.apply_colormap(self.bg_image_array, cmap_name='jet')
                self.bg_image.setImage(self.colored_img, autoLevels=False)  # autoLevels=False for better performance
                self._last_image_hash = current_hash
        else:
            # For grayscale, always update (it's less expensive anyway)
            self.bg_image.setImage(self.bg_image_array, autoLevels=False)  # autoLevels=False for better performance

    def on_polar_plot_clicked(self, event):
        """Handle mouse click event on the polar plot"""
        # Only allow marking when angle display is enabled
        if not self.show_angle:
            return
        
        # Get the position in scene coordinates
        scene_pos = event.scenePos()
        
        # Convert scene coordinates to plot coordinates
        view_box = self.polar_plot.getViewBox()
        plot_pos = view_box.mapSceneToView(scene_pos)
        
        x, y = plot_pos.x(), plot_pos.y()
        
        # Check if click is within the polar plot bounds (-1 to 1)
        if abs(x) <= 1.0 and abs(y) <= 1.0:
            # Store the marker position
            self.reference_markers.append((x, y))
            
            # Create visual marker
            self.create_reference_marker(x, y)
            
            print(f"Reference marker placed at ({x:.3f}, {y:.3f})")

    def create_reference_marker(self, x, y):
        """Create a visual marker for the reference point"""
        # Create a circle marker
        circle = pg.CircleROI([x - 0.05, y - 0.05], [0.1, 0.1], 
                             pen=pg.mkPen('red', width=2), 
                             movable=False, 
                             removable=False)
        circle.handlePen = pg.mkPen('red', width=2)
        circle.handleHoverPen = pg.mkPen('red', width=2)
        
        # Create small crosshairs
        crosshair_length = 0.08
        
        # Vertical crosshair
        v_crosshair = pg.PlotDataItem(
            x=[x, x], 
            y=[y - crosshair_length/2, y + crosshair_length/2],
            pen=pg.mkPen('cyan', width=1)
        )
        
        # Horizontal crosshair
        h_crosshair = pg.PlotDataItem(
            x=[x - crosshair_length/2, x + crosshair_length/2], 
            y=[y, y],
            pen=pg.mkPen('cyan', width=1)
        )
        
        # Add items to the polar plot
        self.polar_plot.addItem(circle)
        self.polar_plot.addItem(v_crosshair)
        self.polar_plot.addItem(h_crosshair)
        
        # Store the marker items for later removal if needed
        marker_group = {
            'circle': circle,
            'v_crosshair': v_crosshair,
            'h_crosshair': h_crosshair,
            'position': (x, y)
        }
        self.reference_marker_items.append(marker_group)

    def clear_reference_markers(self):
        """Clear all reference markers from the polar plot"""
        for marker_group in self.reference_marker_items:
            self.polar_plot.removeItem(marker_group['circle'])
            self.polar_plot.removeItem(marker_group['v_crosshair'])
            self.polar_plot.removeItem(marker_group['h_crosshair'])
        
        self.reference_marker_items.clear()
        self.reference_markers.clear()
        print("All reference markers cleared")

    def toggle_segmentation(self):
        """Toggle segmentation on/off"""
        self.segmentation_enabled = not self.segmentation_enabled
        print(f"Segmentation enabled: {self.segmentation_enabled}")
        
        # Update button color based on state
        if self.segmentation_enabled:
            self.segmentation_btn.setStyleSheet("background-color: #4CAF50; color: white; font-size: 25px; border-radius: 10px; padding: 10px;")
        else:
            self.segmentation_btn.setStyleSheet("background-color: #F44336; color: white; font-size: 25px; border-radius: 10px; padding: 10px;")
            # Remove all highlighted regions
            for region in self.highlighted_regions:
                self.plot_widget.removeItem(region)
            self.highlighted_regions = []
            # Reset segmentation state
            self.current_segmentation_active = False
            self.current_segmentation_intensity = 0.0

    def update_save_data_button_state(self):
        """Update save data button appearance based on save_data and show_angle states"""
        if not self.save_data:
            # State 1: Red - Save data is not active
            self.btn.setStyleSheet("background-color: #F44336; color: white; font-size: 25px; border-radius: 10px; padding: 10px;")
        elif self.save_data and not self.show_angle:
            # State 2: Yellow - Save data is armed but orientation plot is not active
            self.btn.setStyleSheet("background-color: #FFC107; color: black; font-size: 25px; border-radius: 10px; padding: 10px;")
        elif self.save_data and self.show_angle:
            # State 3: Green - Save data is active and orientation plot is active (actually saving)
            self.btn.setStyleSheet("background-color: #4CAF50; color: white; font-size: 25px; border-radius: 10px; padding: 10px;")

    def toggle_angle_display(self):
        """Toggle angle display on/off"""
        self.show_angle = not self.show_angle
        
        # Reset background image arrays
        self.bg_image_array = np.zeros((self.image_dimensions, self.image_dimensions), dtype=float)
        self.colored_img = np.zeros((self.image_dimensions, self.image_dimensions, 3), dtype=np.uint8)
        self.max_intensity_array = np.zeros((self.image_dimensions, self.image_dimensions), dtype=float)
        
        # Reset optimization-related state variables to prevent display issues
        self.polar_update_counter = 0
        self.blur_counter = 0
        if hasattr(self, '_last_image_hash'):
            self._last_image_hash = 0
        
        # Force image update by setting it directly
        self.bg_image.setImage(self.bg_image_array, autoLevels=False)
        
        if self.show_angle:
            self.toggle_angle_btn.setText('Orientation')
            self.toggle_angle_btn.setStyleSheet("background-color: #4CAF50; color: white; font-size: 25px; border-radius: 10px; padding: 10px;")
            self.crosshair_v.setPos(0)
            self.crosshair_h.setPos(0)
            print("Angle display enabled - click on polar plot to place reference markers")
        else:
            self.toggle_angle_btn.setText('Orientation')
            self.toggle_angle_btn.setStyleSheet("background-color: #F44336; color: white; font-size: 25px; border-radius: 10px; padding: 10px;")
            self.crosshair_v.setPos(0)
            self.crosshair_h.setPos(0)
            # Clear all reference markers when angle display is turned off
            self.clear_reference_markers()
        
        # Update save data button state based on new angle display state
        self.update_save_data_button_state()

    def on_annotation_text_changed(self, text):
        """Handle annotation text changes"""
        self.annotation_text = text

    def on_threshold_changed(self, value):
        """Handle threshold changes"""
        self.segmentation_threshold = value
        print(f"Segmentation threshold updated to: {self.segmentation_threshold}")

    def on_front_threshold_changed(self, value):
        """Handle front threshold spinbox changes"""
        self.front_plot_threshold = value
        print(f"Front plot threshold updated to: {self.front_plot_threshold}")
        if hasattr(self, 'distance') and self.distance is not None:
            self.initialize_weighting_array(self.distance)

    def on_y_axis_min_changed(self, text):
        """Handle Y-axis minimum value changes"""
        try:
            self.y_axis_min = int(text)
            print(f"Y-axis minimum updated to: {self.y_axis_min}")
        except ValueError:
            print("Invalid Y-axis minimum value. Please enter a valid integer.")

    def on_y_axis_max_changed(self, text):
        """Handle Y-axis maximum value changes"""
        try:
            self.y_axis_max = int(text)
            print(f"Y-axis maximum updated to: {self.y_axis_max}")
        except ValueError:
            print("Invalid Y-axis maximum value. Please enter a valid integer.")

    def run(self):
        """Run the application"""
        sys.exit(self.app.exec()) 