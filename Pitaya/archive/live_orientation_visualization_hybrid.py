import sys
import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QTimer
import socket
import time
from datetime import datetime
import pickle
import os
import argparse
from scipy.spatial.transform import Rotation as R

# Import from our utils module
from live_orientation_utils import BaseVisualizationMixin

# Parse arguments
parser = argparse.ArgumentParser(description="SoundPass Pitaya Live Guidance Visualization")
parser.add_argument("--pulse-len", type=int, default=6000, help="Number of samples per pulse")
args, unknown = parser.parse_known_args()

PULSE_LEN = args.pulse_len
BUFFER_SIZE = (PULSE_LEN + 9) * 4
PORT = 5005


class TimeSeriesPlotter(BaseVisualizationMixin):
    def __init__(self, screen_resolution="medium"):
        self.app = QApplication(sys.argv)
        
        # Initialize common variables from mixin
        self.initialize_common_variables()
        
        # Configure window sizing for the specified screen resolution
        self.configure_window_sizing(screen_resolution)
        
        # Initialize data structures
        self.x_data = np.arange(PULSE_LEN) * 0.000616 # Red Pitaya timebase -> depth in cm
        self.distance = self.x_data
        
        # Track data acquisition count and session start time
        self.data_acquisition_count = 0
        self.data_acquisition_start_time = None
        
        # Initialize angle tracking
        self.pitch_angles = [0.0]
        self.roll_angles = [0.0]
        self.yaw_angles = [0.0]
        self.initial_quat = None
        
        # Setup GUI
        self.setup_gui()
        
        # Setup connections
        self.setup_connections()
        
        # Initialize weighting array for wavelet processing (from mixin)
        self.initialize_weighting_array(self.distance)
        
        # Create directory for saving files
        now = datetime.now()
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_directory = os.path.join(base_dir, "new_datasets", now.strftime("%Y-%m-%d"))
        os.makedirs(self.save_directory, exist_ok=True)
        
        # Setup TCP Server Socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('0.0.0.0', PORT))
        self.server_socket.listen(1)
        self.server_socket.setblocking(False) # Non-blocking for the UI thread
        self.conn = None
        self.recv_buffer = bytearray()
        
        print(f"TCP Server listening on port {PORT} (expecting PULSE_LEN={PULSE_LEN})...")
        
        # Setup timer for non-blocking read & plot updates (attempt ~100Hz updates)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(10)
        
        # Setup main timer for Mixin refresh rate tracking
        self.main_timer = QTimer()
        self.main_timer.timeout.connect(self.main_update)
        self.main_timer.start(50)  # 20Hz refresh tracking

    def setup_gui(self):
        """Setup the GUI with all controls"""
        self.w = QtWidgets.QWidget()
        self.w.setWindowTitle('Pitaya Live Guidance Visualization')
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
        
        self.w.show()

    def setup_connections(self):
        """Setup all signal connections"""
        self.btn.pressed.connect(self.btn_pressed)
        # Setup common connections from mixin
        self.setup_common_connections()

    def update_plot(self):
        """Fetch TCP telemetry and update plot with current sensor data"""
        # 1. Accept client connection
        if self.conn is None:
            try:
                self.conn, addr = self.server_socket.accept()
                self.conn.setblocking(False)
                print(f"Connected by {addr}")
            except BlockingIOError:
                return
            except Exception as e:
                print(f"Accept error: {e}")
                return

        # 2. Read full frame from TCP socket
        try:
            while len(self.recv_buffer) < BUFFER_SIZE:
                try:
                    packet = self.conn.recv(BUFFER_SIZE - len(self.recv_buffer))
                    if not packet:
                        print("Connection closed by peer.")
                        self.conn.close()
                        self.conn = None
                        self.recv_buffer = bytearray()
                        return
                    self.recv_buffer.extend(packet)
                except BlockingIOError:
                    return # Wait for the next timer event for more data
            
            # Unpack flat floats from network frame
            raw_frame = np.frombuffer(self.recv_buffer, dtype=np.float32).copy()
            self.recv_buffer = bytearray() # Reset buffer for next frame
            
            num_pulses = int(raw_frame[0])
            qw, qx, qy, qz = raw_frame[1:5]
            sys_cal, gyro_cal, accel_cal, mag_cal = [int(v) for v in raw_frame[5:9]]
            avg_pulse = raw_frame[9:]
            
            # 3. Update Plots
            if len(avg_pulse) > self.front_plot_threshold:
                cropped_samples = avg_pulse[self.front_plot_threshold:]
                cropped_distance = self.distance[self.front_plot_threshold:]
                self.plot1.setData(cropped_distance, cropped_samples)
            else:
                self.plot1.setData(self.distance, avg_pulse)
            
            # Process and update wavelet plot (using mixin method)
            wavelet_data, wavelet_distance = self.process_wavelet_data(avg_pulse)
            if wavelet_data is not None and wavelet_distance is not None:
                self.wavelet_plot1.setData(wavelet_distance, wavelet_data)
            
            # Handle segmentation (using mixin method)
            if self.segmentation_enabled:
                self.process_segmentation(wavelet_data, wavelet_distance)
            else:
                self.current_segmentation_active = False
                self.current_segmentation_intensity = 0.0
            
            # 4. Handle Orientation Visualization
            current_orientation_data = None
            if self.show_angle:
                current_orientation_data = self.get_current_orientation_data(qw, qx, qy, qz)
                if current_orientation_data is not None:
                    self.polar_update_counter += 1
                    if self.polar_update_counter >= self.polar_update_frequency:
                        self.polar_update_counter = 0
                        self.update_orientation_display(avg_pulse, current_orientation_data)
                    else:
                        self.update_orientation_crosshairs_only(current_orientation_data)
            
            # Update elapsed time if data acquisition is active
            if self.save_data and self.data_acquisition_start_time is not None:
                elapsed_time = time.time() - self.data_acquisition_start_time
                minutes = int(elapsed_time // 60)
                seconds = int(elapsed_time % 60)
                self.elapsed_time_display.setText(f"Elapsed Time: {minutes}m {seconds}s")
            
            # 5. Save Data if armed
            if self.save_data:
                self.save_current_data(avg_pulse, current_orientation_data)

        except Exception as e:
            print(f"Error in update_plot: {e}")
            if self.conn:
                self.conn.close()
                self.conn = None
            self.recv_buffer = bytearray()

    def get_current_orientation_data(self, qw, qx, qy, qz):
        """Process raw BNO055 quaternion into relative Euler angles"""
        try:
            quat_array = np.array([qx, qy, qz, qw]) # scipy expects [x, y, z, w]
            norm = np.linalg.norm(quat_array)
            
            if abs(norm - 1.0) >= 0.1:
                return None
                
            current_quat = R.from_quat(quat_array)
            
            if self.initial_quat is None:
                print("Setting initial quaternion reference baseline...")
                self.initial_quat = current_quat
                return None
            
            # Compute relative rotation (from initial reference baseline to current position)
            relative_quat = current_quat * self.initial_quat.inv()
            relative_euler = relative_quat.as_euler('xyz', degrees=True)
            relative_pitch = -relative_euler[0]
            relative_roll = relative_euler[1]
            relative_yaw = relative_euler[2]
            
            return [relative_pitch, relative_roll, relative_yaw]
            
        except Exception as e:
            print(f"Error processing orientation: {e}")
            return None

    def update_orientation_display(self, avg_pulse, current_orientation_data):
        """Update polar plot tracking visual overlays"""
        try:
            relative_pitch, relative_roll, relative_yaw = current_orientation_data
            
            self.pitch_angles.append(relative_pitch)
            self.roll_angles.append(relative_roll)
            self.yaw_angles.append(relative_yaw)

            if len(self.pitch_angles) > 10:
                self.pitch_angles.pop(0)
                self.roll_angles.pop(0)
                self.yaw_angles.pop(0)

            # Map configured Euler angles to polar coordinate space
            angles_dict = {'Roll': relative_roll, 'Pitch': relative_pitch, 'Yaw': relative_yaw}
            x_val = angles_dict[self.x_axis_combo.currentText()]
            y_val = angles_dict[self.y_axis_combo.currentText()]
            
            # Calculate tip position using mapped axis values
            x, y = self.calculate_tip_position(x_val, y_val)
            self.update_crosshairs(x, y)
            
            if self.segmentation_enabled:
                ix, iy = self.plot_coords_to_image_indices(-y, -x)
                self.update_image_region(ix, iy, self.current_segmentation_intensity)
                self.update_polar_image()
                
        except Exception as e:
            print(f"Error updating orientation display: {e}")

    def update_orientation_crosshairs_only(self, current_orientation_data):
        """Smooth crosshair update by skipping heavy polar rendering routines"""
        try:
            relative_pitch, relative_roll, relative_yaw = current_orientation_data
            
            angles_dict = {'Roll': relative_roll, 'Pitch': relative_pitch, 'Yaw': relative_yaw}
            x_val = angles_dict[self.x_axis_combo.currentText()]
            y_val = angles_dict[self.y_axis_combo.currentText()]
            
            x, y = self.calculate_tip_position(x_val, y_val)
            self.update_crosshairs(x, y)
        except Exception as e:
            print(f"Error updating crosshairs: {e}")

    def save_current_data(self, avg_pulse, current_orientation_data):
        """Save a data frame containing averaged signal and orientation metrics"""
        try:
            if self.save_data and hasattr(self, 'sub_directory'):
                formatted_date_time = int(time.time() * 1000)
                time_since_segmentation_start = int(time.time() - self.data_acquisition_start_time) if self.data_acquisition_start_time else 0
                file_name = f"{self.sub_directory}/{formatted_date_time}_time{time_since_segmentation_start}.pkl"
                
                orientation_to_save = current_orientation_data if current_orientation_data is not None else [0.0, 0.0, 0.0]
                
                # We wrap avg_pulse in np.array([avg_pulse]) to yield a shape of (1, PULSE_LEN).
                # This guarantees 100% backward-compatibility with Picoscope playback tools,
                # which expect a 2D array representing raw buffer sweeps.
                with open(file_name, 'wb') as file:
                    pickle.dump([np.array([avg_pulse]), self.distance, orientation_to_save], file)
        except Exception as e:
            print(f"Error saving data: {e}")

    def btn_pressed(self):
        """Toggle recording state and allocate folder directories"""
        self.save_data = not self.save_data
        if self.save_data:
            # Count the number of subdirectories to determine sample number
            sub_dirs = [name for name in os.listdir(self.save_directory) if os.path.isdir(os.path.join(self.save_directory, name))]
            sub_directory_count = len(sub_dirs) + 1
            self.data_acquisition_count_display.setText(f"Sample Number: {sub_directory_count}")

            self.data_acquisition_start_time = time.time()

            print(f'Starting to save data...')
            sub_directory = os.path.join(self.save_directory, str(int(time.time())))
            sub_directory += '_' + self.annotation_text + str(sub_directory_count)
            os.makedirs(sub_directory, exist_ok=True)
            print(f'Created sub-directory: {sub_directory}')
            self.sub_directory = sub_directory
        else:
            print(f'Stopped saving data.')
        
        self.update_save_data_button_state()

    def toggle_angle_display(self):
        """Toggle orientation display and reset baseline quaternion"""
        super().toggle_angle_display()
        if self.show_angle:
            self.initial_quat = None # Reset baseline so it captures the next valid frame as reference

    def closeEvent(self, event):
        if self.conn:
            self.conn.close()
        self.server_socket.close()


if __name__ == '__main__':
    plotter = TimeSeriesPlotter(screen_resolution="medium")
    plotter.run()
