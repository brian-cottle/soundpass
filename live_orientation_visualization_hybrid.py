import sys
import numpy as np
from PyQt6.QtWidgets import QApplication
from PyQt6 import QtWidgets
from PyQt6.QtCore import QTimer
import ctypes
from picosdk.ps3000a import ps3000a as ps
import numpy as np
from picosdk.functions import adc2mV, mV2adc, assert_pico_ok
import time
import board
import busio
import adafruit_bno055
import numpy as np
from datetime import datetime
import pickle
import os
from scipy.spatial.transform import Rotation as R

# Import from our utils module
from live_orientation_utils import BaseVisualizationMixin

def setup_IMU():
    i2c = busio.I2C(board.SCL, board.SDA)
    sensor = adafruit_bno055.BNO055_I2C(i2c)
    sensor.axis_remap = (0,1,2,0,1,1)
    return(sensor)

class TimeSeriesPlotter(BaseVisualizationMixin):
    def __init__(self, num_buffers=20, screen_resolution="small"):
        self.app = QApplication(sys.argv)
        
        # Initialize common variables from mixin
        self.initialize_common_variables()
        
        # Configure window sizing for the specified screen resolution
        self.configure_window_sizing(screen_resolution)
        
        # Initialize data structures
        self.x_data = []
        self.y_data = []
        self.y2_data = []
        
        # Track data acquisition count and session start time
        self.data_acquisition_count = 0
        self.data_acquisition_start_time = None
        
        # Initialize angle tracking
        self.pitch_angles = [0]
        self.roll_angles = [0]
        self.yaw_angles = [0]
        
        # Add IMU error handling variables
        self.last_valid_quat = None
        self.imu_error_count = 0
        self.max_imu_errors = 5
        self.imu_retry_delay = 0.1
        
        # Configure number of buffers for data collection
        self.num_buffers = num_buffers
        
        # Setup GUI
        self.setup_gui()
        
        # Setup connections
        self.setup_connections()
        
        # Setup sensor
        self.sensor = setup_IMU()
        
        # Create directory for saving files
        now = datetime.now()
        self.save_directory = "/home/briancottle/Code/SoundPass/new_datasets/" + now.strftime("%Y-%m-%d")
        os.makedirs(self.save_directory, exist_ok=True)
        
        # Setup PicoScope
        self.setup_picoscope()
        
        # Setup timers
        self.main_timer = QTimer()
        self.main_timer.timeout.connect(self.main_update)
        self.main_timer.start(5)  # 20Hz for GUI updates

    def setup_gui(self):
        """Setup the GUI with all controls"""
        self.w = QtWidgets.QWidget()
        self.w.setWindowTitle('Live Data Visualization')
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

    def setup_picoscope(self):
        """Setup PicoScope for data acquisition"""
        # Initialize PicoScope parameters
        self.preTriggerSamples = 0
        self.postTriggerSamples = 7500 # default is 30000, target is 7500
        self.timebase = 4 # default is 2, target is 4
        self.nChannelProperties = 1
        self.autoTriggerMilliseconds = 10000
        self.chARange = 6
        self.advanced_trigger_threshold = 500
        self.sig_gen_peak_to_peak = 2000000
        self.sig_gen_hz = 1000
        self.sig_gen_offset = 00000000
        
        # Create distance array
        scaling_factor = 1 # default is 4, target is 1
        self.distance = np.arange((self.preTriggerSamples + self.postTriggerSamples))/(scaling_factor*1000)*13/10  
        self.x_data = self.distance
        
        # Initialize weighting array for wavelet processing (from mixin)
        self.initialize_weighting_array(self.distance)
        
        # Initialize PicoScope
        self.status = {}
        self.chandle = ctypes.c_int16()
        
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

        # Creates a overlow location for data
        self.overflow = (ctypes.c_int16 * self.num_buffers)()
        # Creates converted types maxsamples
        self.cmaxSamples = ctypes.c_int32(self.maxsamples)

        self.status["MemorySegments"] = ps.ps3000aMemorySegments(self.chandle, self.num_buffers, ctypes.byref(self.cmaxSamples))
        assert_pico_ok(self.status["MemorySegments"])

        # sets number of captures
        self.status["SetNoOfCaptures"] = ps.ps3000aSetNoOfCaptures(self.chandle, self.num_buffers)
        assert_pico_ok(self.status["SetNoOfCaptures"])

        # Setup data buffers for multiple segments
        self.setup_data_buffers()
        
        # Start the update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(1)

    def setup_data_buffers(self):
        """Setup data buffers for all segments"""
        # Create all buffer arrays
        self.buffers = []
        for i in range(self.num_buffers):
            buffer_max = np.empty(self.maxsamples, dtype=np.dtype('int16'))
            buffer_min = np.empty(self.maxsamples, dtype=np.dtype('int16'))
            self.buffers.append((buffer_max, buffer_min))
            
            # Set data buffers
            self.status[f"SetDataBuffers_{i}"] = ps.ps3000aSetDataBuffers(
                self.chandle, 0, 
                buffer_max.ctypes.data, 
                buffer_min.ctypes.data, 
                self.maxsamples, i, 0)
            assert_pico_ok(self.status[f"SetDataBuffers_{i}"])

    def get_valid_quaternion(self):
        """Safely get a valid quaternion from the IMU sensor with error handling"""
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                # Get quaternion from sensor
                raw_quat = self.sensor.quaternion
                
                # Check if quaternion data is valid
                if raw_quat is None:
                    print(f"Attempt {attempt + 1}: Quaternion is None")
                    time.sleep(self.imu_retry_delay)
                    continue
                
                # Convert to numpy array for easier validation
                quat_array = np.array(raw_quat)
                
                # Check for invalid values (NaN, inf, or all zeros)
                if np.any(np.isnan(quat_array)) or np.any(np.isinf(quat_array)):
                    print(f"Attempt {attempt + 1}: Quaternion contains NaN or inf values: {quat_array}")
                    time.sleep(self.imu_retry_delay)
                    continue
                
                # Check if quaternion norm is too small (effectively zero)
                quat_norm = np.linalg.norm(quat_array)
                if quat_norm < 1e-6:
                    print(f"Attempt {attempt + 1}: Quaternion norm too small: {quat_norm}, values: {quat_array}")
                    time.sleep(self.imu_retry_delay)
                    continue
                
                # Try to create rotation object to validate quaternion
                try:
                    current_quat = R.from_quat(raw_quat)
                    # If successful, store as last valid quaternion
                    self.last_valid_quat = current_quat
                    self.imu_error_count = 0  # Reset error count on success
                    return current_quat
                except ValueError as e:
                    print(f"Attempt {attempt + 1}: Invalid quaternion for rotation: {e}, values: {quat_array}")
                    time.sleep(self.imu_retry_delay)
                    continue
                    
            except Exception as e:
                print(f"Attempt {attempt + 1}: Error reading from IMU: {e}")
                time.sleep(self.imu_retry_delay)
                continue
        
        # All attempts failed
        self.imu_error_count += 1
        print(f"Failed to get valid quaternion after {max_attempts} attempts. Error count: {self.imu_error_count}")
        
        # If we have too many consecutive errors, try to reinitialize IMU
        if self.imu_error_count >= self.max_imu_errors:
            print("Too many IMU errors. Attempting to reinitialize sensor...")
            try:
                self.sensor = setup_IMU()
                self.imu_error_count = 0
                print("IMU reinitialized successfully")
                # Try to get initial quaternion after reset
                if hasattr(self, 'initial_quat'):
                    time.sleep(0.2)  # Give sensor time to stabilize
                    return self.get_valid_quaternion()  # Recursive call after reset
            except Exception as e:
                print(f"Failed to reinitialize IMU: {e}")
        
        # Return last valid quaternion if available, otherwise None
        if self.last_valid_quat is not None:
            print("Using last valid quaternion as fallback")
            return self.last_valid_quat
        
        return None

    def update_plot(self):
        """Update plot with current sensor data"""
        try:
            # Start data collection
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

            # Wait for data collection to finish
            ready = ctypes.c_int16(0)
            check = ctypes.c_int16(0)
            while ready.value == check.value:
                self.status["isReady"] = ps.ps3000aIsReady(self.chandle, ctypes.byref(ready))

            # Get bulk data
            self.status["GetValuesBulk"] = ps.ps3000aGetValuesBulk(self.chandle, ctypes.byref(self.cmaxSamples), 0, self.num_buffers-1, 1, 0, ctypes.byref(self.overflow))
            assert_pico_ok(self.status["GetValuesBulk"])

            # Collect all samples
            all_samples = []
            for i in range(self.num_buffers):  # Use all available buffers
                all_samples.append(self.buffers[i][0])
            
            all_samples = np.asarray(all_samples)
            average_samples = np.mean(all_samples, axis=0)
            
            # Update signal plot - crop data based on front_plot_threshold
            if len(average_samples) > self.front_plot_threshold:
                cropped_samples = average_samples[self.front_plot_threshold:]
                cropped_distance = self.distance[self.front_plot_threshold:]
                self.plot1.setData(cropped_distance, cropped_samples)
            else:
                self.plot1.setData(self.distance, average_samples)
            
            # Process and update wavelet plot (using mixin method)
            wavelet_data, wavelet_distance = self.process_wavelet_data(average_samples)
            if wavelet_data is not None and wavelet_distance is not None:
                self.wavelet_plot1.setData(wavelet_distance, wavelet_data)
            
            # Handle segmentation (using mixin method)
            if self.segmentation_enabled:
                self.process_segmentation(wavelet_data, wavelet_distance)
            else:
                # Reset segmentation state when segmentation is disabled
                self.current_segmentation_active = False
                self.current_segmentation_intensity = 0.0
            
            # Handle orientation visualization with frame skipping for performance
            current_orientation_data = None
            if self.show_angle:
                # Get orientation data for both display and saving (every frame)
                current_orientation_data = self.get_current_orientation_data()
                
                # Only update polar plot every few frames to improve performance
                self.polar_update_counter += 1
                if self.polar_update_counter >= self.polar_update_frequency:
                    self.polar_update_counter = 0
                    self.update_orientation_display(all_samples, average_samples, current_orientation_data)
                else:
                    # Still update crosshairs for smooth movement, just skip the expensive polar plot updates
                    self.update_orientation_crosshairs_only()
            
            # Update elapsed time if data acquisition is enabled
            if self.save_data and self.data_acquisition_start_time is not None:
                elapsed_time = time.time() - self.data_acquisition_start_time
                minutes = int(elapsed_time // 60)
                seconds = int(elapsed_time % 60)
                self.elapsed_time_display.setText(f"Elapsed Time: {minutes}m {seconds}s")
            
            # Save data if enabled (every frame, regardless of frame skipping)
            if self.save_data:
                self.save_current_data(all_samples, current_orientation_data)
                
        except Exception as e:
            print(f"Error in update_plot: {e}")

    def update_orientation_display(self, all_samples, average_samples, current_orientation_data):
        """Update orientation visualization"""
        try:
            # Use passed orientation data if available, otherwise calculate it
            if current_orientation_data is not None:
                relative_pitch, relative_roll, relative_yaw = current_orientation_data
            else:
                # Get current orientation as quaternion from the sensor using robust method
                current_quat = self.get_valid_quaternion()
                
                if current_quat is None:
                    print("Warning: Could not get valid quaternion data. Skipping orientation update.")
                    return

                # Ensure we have a valid initial quaternion
                if not hasattr(self, 'initial_quat') or self.initial_quat is None:
                    print("Setting initial quaternion reference...")
                    self.initial_quat = current_quat
                    return

                # Compute relative rotation (from initial to current)
                relative_quat = current_quat * self.initial_quat.inv()

                # Get relative Euler angles (in device's frame)
                relative_euler = relative_quat.as_euler('xyz', degrees=True)
                relative_roll = relative_euler[0]
                relative_pitch = relative_euler[1]
                relative_yaw = relative_euler[2]

            # Check for sudden changes in pitch or yaw (possible IMU glitch)
            if abs(relative_pitch - self.pitch_angles[-1]) > 45 or abs(relative_yaw - self.yaw_angles[-1]) > 45:
                print(f"Sudden change detected: pitch={relative_pitch:.1f}° (was {self.pitch_angles[-1]:.1f}°), yaw={relative_yaw:.1f}° (was {self.yaw_angles[-1]:.1f}°)")
                print("Resetting IMU reference...")
                
                # Try to get a new stable reference
                stable_quat = self.get_valid_quaternion()
                if stable_quat is not None:
                    self.initial_quat = stable_quat
                    print("IMU reference reset successfully")
                else:
                    print("Failed to get stable quaternion for reset")
                return
            else:
                # Update pitch, roll, and yaw angles
                self.pitch_angles.append(relative_pitch)
                self.roll_angles.append(relative_roll)
                self.yaw_angles.append(relative_yaw)

                if len(self.pitch_angles) > 10:
                    self.pitch_angles.pop(0)
                    self.roll_angles.pop(0)
                    self.yaw_angles.pop(0)

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

    def update_orientation_crosshairs_only(self):
        """Update only the crosshairs without expensive polar plot operations"""
        try:
            # Get current orientation as quaternion from the sensor using robust method
            current_quat = self.get_valid_quaternion()
            
            if current_quat is None:
                return

            # Ensure we have a valid initial quaternion
            if not hasattr(self, 'initial_quat') or self.initial_quat is None:
                return

            # Compute relative rotation (from initial to current)
            relative_quat = current_quat * self.initial_quat.inv()

            # Get relative Euler angles (in device's frame)
            relative_euler = relative_quat.as_euler('xyz', degrees=True)
            relative_pitch = relative_euler[1]
            relative_yaw = relative_euler[2]

            # Calculate tip position (using mixin method)
            x, y = self.calculate_tip_position(relative_pitch, relative_yaw)
            
            # Update crosshairs (using mixin method)
            self.update_crosshairs(x, y)
            
        except Exception as e:
            print(f"Error updating crosshairs: {e}")

    def get_current_orientation_data(self):
        """Get current orientation data without expensive display operations"""
        try:
            # Get current orientation as quaternion from the sensor using robust method
            current_quat = self.get_valid_quaternion()
            
            if current_quat is None:
                return None

            # Ensure we have a valid initial quaternion
            if not hasattr(self, 'initial_quat') or self.initial_quat is None:
                return None

            # Compute relative rotation (from initial to current)
            relative_quat = current_quat * self.initial_quat.inv()

            # Get relative Euler angles (in device's frame)
            relative_euler = relative_quat.as_euler('xyz', degrees=True)
            relative_roll = relative_euler[0]
            relative_pitch = relative_euler[1]
            relative_yaw = relative_euler[2]

            return [relative_pitch, relative_roll, relative_yaw]
            
        except Exception as e:
            print(f"Error getting orientation data: {e}")
            return None

    def save_current_data(self, all_samples, current_orientation_data):
        """Save current data to file every frame"""
        try:
            if self.save_data and hasattr(self, 'sub_directory'):
                now = datetime.now()
                formatted_date_time = int(time.time() * 1000)
                time_since_segmentation_start = int(time.time() - self.data_acquisition_start_time) if self.data_acquisition_start_time else 0
                file_name = f"{self.sub_directory}/{formatted_date_time}_time{time_since_segmentation_start}.pkl"
                
                # Use orientation data if available, otherwise use default values
                orientation_to_save = current_orientation_data if current_orientation_data is not None else [0, 0, 0]
                
                with open(file_name, 'wb') as file:
                    pickle.dump([all_samples, self.distance, orientation_to_save], file)
        except Exception as e:
            print(f"Error saving data: {e}")

    def btn_pressed(self):
        """Handle data acquisition button press"""
        self.save_data = not self.save_data
        if self.save_data:
            # Count the number of subdirectories in the save directory
            sub_directory_count = len([name for name in os.listdir(self.save_directory) if os.path.isdir(os.path.join(self.save_directory, name))]) + 1
            self.data_acquisition_count_display.setText(f"Sample Number: {sub_directory_count}")

            # Record the start time of the data acquisition session
            self.data_acquisition_start_time = time.time()

            print(f'Starting to save data')
            sub_directory = os.path.join(self.save_directory, str(int(time.time())))
            sub_directory += '_' + self.annotation_text + str(sub_directory_count)
            os.makedirs(sub_directory, exist_ok=True)
            print(f'Created sub-directory: {sub_directory}')
            self.sub_directory = sub_directory
        else:
            print(f'Not saving data')
        
        # Update button state using the new three-state logic
        self.update_save_data_button_state()

    def toggle_angle_display(self):
        """Toggle angle display on/off - override to add IMU initialization"""
        # Call the parent method from mixin
        super().toggle_angle_display()
        
        # Add IMU-specific initialization
        if self.show_angle:
            # Initialize IMU reference using robust method
            initial_quat = self.get_valid_quaternion()
            if initial_quat is not None:
                self.initial_quat = initial_quat
                print("IMU reference initialized successfully")
            else:
                print("Warning: Could not initialize IMU reference quaternion")

    def on_y_axis_min_changed(self, text):
        """Handle Y-axis minimum value changes and update plot"""
        try:
            self.y_axis_min = int(text)
            self.plot_widget.setYRange(self.y_axis_min, self.y_axis_max)
            print(f"Y-axis minimum updated to: {self.y_axis_min}")
        except ValueError:
            print("Invalid Y-axis minimum value. Please enter a valid integer.")

    def on_y_axis_max_changed(self, text):
        """Handle Y-axis maximum value changes and update plot"""
        try:
            self.y_axis_max = int(text)
            self.plot_widget.setYRange(self.y_axis_min, self.y_axis_max)
            print(f"Y-axis maximum updated to: {self.y_axis_max}")
        except ValueError:
            print("Invalid Y-axis maximum value. Please enter a valid integer.")

    def run(self):
        """Run the application"""
        sys.exit(self.app.exec())

if __name__ == '__main__':
    # You can configure the number of buffers and screen resolution here
    # Screen resolution options: "small", "medium", "large"
    # For small screens (1024x768, 1366x768): use "small"
    # For medium screens (1440x900, 1600x900): use "medium" 
    # For large screens (1920x1080, 2560x1440): use "large"
    plotter = TimeSeriesPlotter(num_buffers=80, screen_resolution="medium")
    plotter.run() 