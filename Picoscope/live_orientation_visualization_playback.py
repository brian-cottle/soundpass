import sys
import numpy as np
from PyQt6.QtWidgets import QApplication, QSlider, QFileDialog
from PyQt6 import QtWidgets
from PyQt6.QtCore import QTimer
import pyqtgraph as pg
import time
from glob import glob
from datetime import datetime
import pickle
from PyQt6.QtCore import Qt
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

# Import from our utils module
from live_orientation_utils import BaseVisualizationMixin, low_pass_filter



class TimeSeriesPlayback(BaseVisualizationMixin):
    def __init__(self, data_directory=None, screen_resolution="medium"):
        self.app = QApplication(sys.argv)
        self.data_directory = data_directory
        self.current_file_index = 0
        self.is_playing = False
        
        # Initialize data structures
        self.pickle_files = []
        self.distance = None
        
        if data_directory:
            self.load_data_directory(data_directory)
        
        # Setup video writer
        self.frame_size = (1920, 1080)
        self.video_writer = None
        self.output_video_path = None
        self.recording_video = False
        
        # Initialize common variables from mixin
        self.initialize_common_variables()
        
        # Configure window sizing for the specified screen resolution
        self.configure_window_sizing(screen_resolution)
        
        # Setup GUI
        self.setup_gui()
        
        # Setup timers
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self.update_plot)
        
        self.main_timer = QTimer()
        self.main_timer.timeout.connect(self.main_update)
        self.main_timer.start(50)  # 20Hz for GUI updates

    def load_data_directory(self, data_directory):
        """Load all pickle files from the specified directory"""
        self.pickle_files = natsorted(glob(os.path.join(data_directory, "*.pkl")))
        if not self.pickle_files:
            print(f"No pickle files found in {data_directory}")
            return
        
        print(f"Found {len(self.pickle_files)} pickle files")
        
        # Load first file to get distance array
        if self.pickle_files:
            try:
                with open(self.pickle_files[0], 'rb') as f:
                    data = pickle.load(f)
                    self.distance = data[1]
                    print(f"Loaded distance array with {len(self.distance)} points")
                    
                    # Initialize weighting array for wavelet processing (from mixin)
                    self.initialize_weighting_array(self.distance)
            except Exception as e:
                print(f"Error loading first file: {e}")



    def setup_gui(self):
        """Setup the GUI with all controls"""
        self.w = QtWidgets.QWidget()
        self.w.setWindowTitle('Data Playback Visualization')
        layout = QtWidgets.QGridLayout()
        self.w.setLayout(layout)
        
        # Setup window sizing
        self.setup_window_sizing(self.w)
        
        # Left panel controls (from mixin)
        self.setup_left_controls(layout)
        
        # Main plot widget (from mixin)
        self.setup_main_plot(layout)
        
        # Right panel - polar plot (from mixin)
        self.setup_polar_plot(layout)
        
        # Playback controls
        self.setup_playback_controls(layout)
        
        # Setup connections
        self.setup_connections()
        
        self.w.show()

    def setup_left_controls_with_playback(self, layout):
        """Setup left panel controls including playback-specific controls"""
        # Call parent method for common controls
        super().setup_left_controls(layout)

    # Remove duplicated methods - inherit from BaseVisualizationMixin instead

    def get_dynamic_button_style(self, background_color):
        """Generate dynamic button styling based on current font size"""
        border_radius = max(5, self.font_size // 2)
        padding = max(5, self.font_size // 2)
        return f"background-color: {background_color}; color: white; font-size: {self.font_size}px; border-radius: {border_radius}px; padding: {padding}px;"

    def setup_playback_controls(self, layout):
        """Setup playback controls at the bottom"""
        # Playback button
        self.playback_btn = QtWidgets.QPushButton('Play')
        self.playback_btn.setFixedSize(self.button_width, self.button_height)
        self.playback_btn.setStyleSheet(self.get_dynamic_button_style("#4CAF50"))
        layout.addWidget(self.playback_btn, 15, 0)
        
        # Directory selection button
        self.dir_btn = QtWidgets.QPushButton('Select Directory')
        self.dir_btn.setFixedSize(self.button_width, self.button_height)
        self.dir_btn.setStyleSheet(self.get_dynamic_button_style("#FF5722"))
        layout.addWidget(self.dir_btn, 16, 0)
        
        # Progress slider
        self.progress_slider = QSlider(Qt.Orientation.Horizontal)
        self.progress_slider.setMinimum(0)
        self.progress_slider.setMaximum(100)
        self.progress_slider.setValue(0)
        self.progress_slider.setFixedHeight(self.control_height)
        layout.addWidget(self.progress_slider, 16, 2, 1, 3)
        
        # Progress label
        self.progress_label = QtWidgets.QLabel("0 / 0")
        self.progress_label.setFixedHeight(self.control_height)
        self.progress_label.setStyleSheet(f"font-size: {self.font_size}px;")
        layout.addWidget(self.progress_label, 17, 2, 1, 3)

    def setup_connections(self):
        """Setup all signal connections"""
        self.btn.pressed.connect(self.btn_pressed)
        # Setup common connections from mixin
        self.setup_common_connections()
        # Playback-specific connections
        self.playback_btn.pressed.connect(self.toggle_playback)
        self.dir_btn.pressed.connect(self.select_directory)
        self.progress_slider.valueChanged.connect(self.on_slider_changed)

    def select_directory(self):
        """Select data directory for playback"""
        directory = QFileDialog.getExistingDirectory(self.w, "Select Data Directory")
        if directory:
            self.data_directory = directory
            self.load_data_directory(directory)
            self.current_file_index = 0
            self.update_progress_slider()
            print(f"Selected directory: {directory}")

    def toggle_playback(self):
        """Toggle playback on/off"""
        if not self.pickle_files:
            print("No data loaded. Please select a directory first.")
            return
            
        self.is_playing = not self.is_playing
        
        if self.is_playing:
            self.playback_btn.setText('Pause')
            self.playback_btn.setStyleSheet(self.get_dynamic_button_style("#FF5722"))
            self.setup_video_writer()
            self.recording_video = True
            self.playback_timer.start(143)  # 7Hz to match the data acquisition rate
        else:
            self.playback_btn.setText('Play')
            self.playback_btn.setStyleSheet(self.get_dynamic_button_style("#4CAF50"))
            self.playback_timer.stop()
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
                self.recording_video = False
                print(f"Video saved to: {self.output_video_path}")

    def setup_video_writer(self):
        """Setup video writer for recording"""
        if self.data_directory:
            self.output_video_path = os.path.join(self.data_directory, "playback_video.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.output_video_path,
                fourcc,
                14.0,  # 14 FPS for smooth video (2x data rate for better visual quality)
                self.frame_size
            )

    def capture_frame(self):
        """Capture current frame for video"""
        if not self.recording_video or not self.video_writer:
            return
            
        try:
            screen = self.w.grab()
            image = screen.toImage()
            width = image.width()
            height = image.height()
            ptr = image.bits()
            ptr.setsize(height * width * 4)
            arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
            frame = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
            frame = cv2.resize(frame, self.frame_size)
            self.video_writer.write(frame)
        except Exception as e:
            print(f"Error capturing frame: {e}")

    def update_progress_slider(self):
        """Update progress slider range and position"""
        if self.pickle_files:
            self.progress_slider.setMaximum(len(self.pickle_files) - 1)
            self.progress_slider.setValue(self.current_file_index)
            self.progress_label.setText(f"{self.current_file_index + 1} / {len(self.pickle_files)}")

    def on_slider_changed(self, value):
        """Handle slider value changes"""
        if not self.is_playing and self.pickle_files:
            self.current_file_index = value
            self.update_progress_slider()
            self.update_plot()

    # main_update method is now inherited from BaseVisualizationMixin

    def update_plot(self):
        """Update plot with current data"""
        if not self.pickle_files or self.current_file_index >= len(self.pickle_files):
            if self.is_playing:
                # End of playback
                self.toggle_playback()
            return
        
        # Load current file
        try:
            with open(self.pickle_files[self.current_file_index], 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"Error loading file {self.pickle_files[self.current_file_index]}: {e}")
            return
        
        # Unpack data
        all_samples = data[0]
        distance = data[1]
        
        # Handle orientation data (may not exist in older files)
        if len(data) > 2:
            orientation = data[2]  # [relative_pitch, relative_roll, relative_yaw]
        else:
            orientation = [0, 0, 0]  # Default values
        
        # Update signal plot
        average_samples = np.mean(all_samples, axis=0)
        
        # Update main plot - crop data based on front_plot_threshold
        if self.distance is not None:
            if len(average_samples) > self.front_plot_threshold:
                cropped_samples = average_samples[self.front_plot_threshold:]
                cropped_distance = self.distance[self.front_plot_threshold:]
                
                # Apply downsampling if enabled
                if self.enable_plot_downsampling:
                    from live_orientation_utils import downsample_data
                    plot_distance, plot_samples = downsample_data(
                        cropped_distance, cropped_samples, self.plot_downsample_factor
                    )
                    self.plot1.setData(plot_distance, plot_samples)
                else:
                    self.plot1.setData(cropped_distance, cropped_samples)
            else:
                # Apply downsampling if enabled
                if self.enable_plot_downsampling:
                    from live_orientation_utils import downsample_data
                    plot_distance, plot_samples = downsample_data(
                        self.distance, average_samples, self.plot_downsample_factor
                    )
                    self.plot1.setData(plot_distance, plot_samples)
                else:
                    self.plot1.setData(self.distance, average_samples)
        
        # Process and update wavelet plot (using mixin method)
        wavelet_data, wavelet_distance = self.process_wavelet_data(average_samples)
        if wavelet_data is not None and wavelet_distance is not None:
            # Apply downsampling if enabled
            if self.enable_plot_downsampling:
                from live_orientation_utils import downsample_data
                plot_wavelet_distance, plot_wavelet_data = downsample_data(
                    wavelet_distance, wavelet_data, self.plot_downsample_factor
                )
                self.wavelet_plot1.setData(plot_wavelet_distance, plot_wavelet_data)
            else:
                self.wavelet_plot1.setData(wavelet_distance, wavelet_data)
        
        # Handle segmentation (using mixin method)
        if self.segmentation_enabled:
            # Use downsampled data for segmentation if downsampling is enabled
            if self.enable_plot_downsampling and wavelet_data is not None and wavelet_distance is not None:
                from live_orientation_utils import downsample_data
                segmentation_distance, segmentation_data = downsample_data(
                    wavelet_distance, wavelet_data, self.plot_downsample_factor
                )
                self.process_segmentation(segmentation_data, segmentation_distance)
            else:
                self.process_segmentation(wavelet_data, wavelet_distance)
        else:
            # Reset segmentation state when segmentation is disabled
            self.current_segmentation_active = False
            self.current_segmentation_intensity = 0.0
        
        # Handle orientation visualization with frame skipping for performance
        if self.show_angle and len(orientation) == 3:
            # Only update polar plot every few frames to improve performance
            self.polar_update_counter += 1
            if self.polar_update_counter >= self.polar_update_frequency:
                self.polar_update_counter = 0
                self.update_orientation_display(orientation)
            else:
                # Still update crosshairs for smooth movement, just skip the expensive polar plot updates
                self.update_orientation_crosshairs_only(orientation)
        
        # Capture frame for video
        if self.recording_video:
            self.capture_frame()
        
        # Update progress
        self.update_progress_slider()
        
        # Move to next frame if playing
        if self.is_playing:
            self.current_file_index += 1

    def update_orientation_display(self, orientation):
        """Update orientation visualization for playback data"""
        try:
            relative_pitch, relative_roll, relative_yaw = orientation
            
            # Calculate tip position (using mixin method)
            x, y = self.calculate_tip_position(relative_pitch, relative_yaw)
            
            # Update crosshairs (using mixin method)
            self.update_crosshairs(x, y)
            
            # Only update background image if segmentation is enabled
            if self.segmentation_enabled:
                # Get coordinates for the tip position
                ix, iy = self.plot_coords_to_image_indices(-y, -x)
                
                # Update a 5x5 area with normalized distribution centered on the crosshair
                self.update_image_region(ix, iy, self.current_segmentation_intensity)
                
                # Apply colormap or grayscale (using mixin method)
                self.update_polar_image()
            
        except Exception as e:
            print(f"Error updating orientation display: {e}")

    def update_orientation_crosshairs_only(self, orientation):
        """Update only the crosshairs without expensive polar plot operations"""
        try:
            relative_pitch, relative_roll, relative_yaw = orientation
            
            # Calculate tip position (using mixin method)
            x, y = self.calculate_tip_position(relative_pitch, relative_yaw)
            
            # Update crosshairs (using mixin method)
            self.update_crosshairs(x, y)
            
        except Exception as e:
            print(f"Error updating crosshairs: {e}")

    def btn_pressed(self):
        """Handle data acquisition button press"""
        self.save_data = not self.save_data
        print(f"Data acquisition: {'ON' if self.save_data else 'OFF'}")
        
        # Update button state using the new three-state logic
        self.update_save_data_button_state()

    def run(self):
        """Run the application"""
        sys.exit(self.app.exec())

if __name__ == '__main__':
    # You can specify a default directory here or leave it None to select via GUI
    default_directory = None  # Example: "/home/briancottle/Code/SoundPass/new_datasets/2025-05-30/1748623248_12"
    
    # You can configure the screen resolution here
    # Screen resolution options: "small", "medium", "large"
    # For small screens (1024x768, 1366x768): use "small"
    # For medium screens (1440x900, 1600x900): use "medium" 
    # For large screens (1920x1080, 2560x1440): use "large"
    playback = TimeSeriesPlayback(default_directory, screen_resolution="medium")
    playback.run() 