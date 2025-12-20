import sys
import numpy as np
import ctypes
from picosdk.ps3000a import ps3000a as ps
import numpy as np
from picosdk.functions import adc2mV, mV2adc, assert_pico_ok
import time
import numpy as np
from datetime import datetime
import pickle
import os
from scipy.spatial.transform import Rotation as R
import signal
import threading

# Import from our utils module
from live_orientation_utils import BaseVisualizationMixin

class MockIMU:
    """Mock IMU sensor for testing when hardware is not available"""
    def __init__(self):
        self.time_start = time.time()
        
    @property
    def quaternion(self):
        # Generate a slowly changing quaternion for testing
        t = time.time() - self.time_start
        # Create a slow rotation around the Z axis
        angle = t * 0.1  # 0.1 radians per second
        quat = R.from_euler('z', angle).as_quat()
        return quat

def setup_IMU():
    """Setup IMU with fallback to mock sensor"""
    try:
        import board
        import busio
        import adafruit_bno055
        
        i2c = busio.I2C(board.SCL, board.SDA)
        sensor = adafruit_bno055.BNO055_I2C(i2c)
        sensor.axis_remap = (0,1,2,0,1,1)
        print("Real IMU sensor initialized")
        return sensor
    except Exception as e:
        print(f"Real IMU not available ({e}), using mock sensor for testing")
        return MockIMU()

class BackendBenchmark(BaseVisualizationMixin):
    def __init__(self, num_buffers=20):
        print("Initializing Backend Benchmark (No GUI)")
        
        # Initialize common variables from mixin
        self.initialize_common_variables()
        

        # Setup optional processing flags (these are the authoritative settings)
        self.enable_wavelet_processing = True
        self.use_bulk_acquisition = True  # Set to True to use bulk acquisition, False for single buffer
        self.segmentation_enabled = True  # Using same name as mixin for consistency
        self.show_angle = True

        # Other backend-only settings
        self.save_data = False  # Don't save data by default
        
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
        
        # Setup sensor
        self.sensor = setup_IMU()
        
        # Setup PicoScope
        self.setup_picoscope()
        
        # Initialize backend-only visualization variables needed by mixin
        self.image_dimensions = 100
        self.bg_image_array = np.zeros((self.image_dimensions, self.image_dimensions), dtype=float)
        self.max_intensity_array = np.zeros((self.image_dimensions, self.image_dimensions), dtype=float)
        self.colored_img = np.zeros((self.image_dimensions, self.image_dimensions, 3), dtype=np.uint8)
        
        # Performance tracking
        self.iteration_count = 0
        self.start_time = time.time()
        self.process_times = []
        self.refresh_rates = []
        self.last_refresh_time = time.time()
        
        # Detailed timing for each processing section
        self.acquisition_times = []
        self.wavelet_times = []
        self.segmentation_times = []
        self.orientation_times = []
        
        # Initialize IMU reference
        self.initialize_imu_reference()
        
        # Setup graceful shutdown
        self.running = True
        signal.signal(signal.SIGINT, self.signal_handler)
        
        print("Backend Benchmark initialized successfully")
        print("Press Ctrl+C to stop and see final statistics")
        print("=" * 50)

    def signal_handler(self, signum, frame):
        """Handle graceful shutdown"""
        print("\nShutting down...")
        self.running = False

    def initialize_imu_reference(self):
        """Initialize IMU reference quaternion"""
        print("Initializing IMU reference...")
        initial_quat = self.get_valid_quaternion()
        if initial_quat is not None:
            self.initial_quat = initial_quat
            print("IMU reference initialized successfully")
        else:
            print("Warning: Could not initialize IMU reference quaternion")

    def setup_picoscope(self):
        """Setup PicoScope for data acquisition"""
        print("Setting up PicoScope...")
        
        # Initialize PicoScope parameters
        self.preTriggerSamples = 0
        self.postTriggerSamples = 30000
        self.timebase = 2
        self.nChannelProperties = 1
        self.autoTriggerMilliseconds = 10000
        self.chARange = 6
        self.advanced_trigger_threshold = 500
        self.sig_gen_peak_to_peak = 2000000
        self.sig_gen_hz = 1000
        self.sig_gen_offset = 00000000
        
        # Create distance array
        self.distance = np.arange((self.preTriggerSamples + self.postTriggerSamples))/(4*1000)*13/10
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
        
        print("PicoScope setup completed successfully")

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
                    time.sleep(self.imu_retry_delay)
                    continue
                
                # Convert to numpy array for easier validation
                quat_array = np.array(raw_quat)
                
                # Check for invalid values (NaN, inf, or all zeros)
                if np.any(np.isnan(quat_array)) or np.any(np.isinf(quat_array)):
                    time.sleep(self.imu_retry_delay)
                    continue
                
                # Check if quaternion norm is too small (effectively zero)
                quat_norm = np.linalg.norm(quat_array)
                if quat_norm < 1e-6:
                    time.sleep(self.imu_retry_delay)
                    continue
                
                # Try to create rotation object to validate quaternion
                try:
                    current_quat = R.from_quat(raw_quat)
                    # If successful, store as last valid quaternion
                    self.last_valid_quat = current_quat
                    self.imu_error_count = 0  # Reset error count on success
                    return current_quat
                except ValueError:
                    time.sleep(self.imu_retry_delay)
                    continue
                    
            except Exception:
                time.sleep(self.imu_retry_delay)
                continue
        
        # All attempts failed
        self.imu_error_count += 1
        
        # If we have too many consecutive errors, try to reinitialize IMU
        if self.imu_error_count >= self.max_imu_errors:
            try:
                self.sensor = setup_IMU()
                self.imu_error_count = 0
                # Try to get initial quaternion after reset
                if hasattr(self, 'initial_quat'):
                    time.sleep(0.2)  # Give sensor time to stabilize
                    return self.get_valid_quaternion()  # Recursive call after reset
            except Exception:
                pass
        
        # Return last valid quaternion if available, otherwise None
        if self.last_valid_quat is not None:
            return self.last_valid_quat
        
        return None

    def process_iteration(self):
        """Process one complete iteration of data acquisition and computation"""
        iteration_start_time = time.time()
        
        try:
            # ============ DATA ACQUISITION TIMING ============
            acquisition_start_time = time.time()
            
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

            # Optionally use GetValuesBulk and averaging, controlled by self.use_bulk_acquisition
            if getattr(self, "use_bulk_acquisition", True):
                # Get bulk data
                self.status["GetValuesBulk"] = ps.ps3000aGetValuesBulk(self.chandle, ctypes.byref(self.cmaxSamples), 0, self.num_buffers-1, 1, 0, ctypes.byref(self.overflow))
                assert_pico_ok(self.status["GetValuesBulk"])

                # Collect all samples
                all_samples = []
                for i in range(self.num_buffers):
                    all_samples.append(self.buffers[i][0])
                
                all_samples = np.asarray(all_samples)
                average_samples = np.mean(all_samples, axis=0)
            else:
                # Fallback: get only the first buffer (single acquisition)
                average_samples = self.buffers[0][0]
            
            acquisition_time = time.time() - acquisition_start_time
            self.acquisition_times.append(acquisition_time)
            
            # ============ WAVELET PROCESSING TIMING ============
            wavelet_start_time = time.time()
            if getattr(self, "enable_wavelet_processing", True):
                wavelet_data, wavelet_distance = self.process_wavelet_data(average_samples)
            else:
                wavelet_data, wavelet_distance = None, None
            wavelet_time = time.time() - wavelet_start_time
            self.wavelet_times.append(wavelet_time)
            
            # ============ SEGMENTATION PROCESSING TIMING ============
            segmentation_start_time = time.time()
            if getattr(self, "segmentation_enabled", False) and wavelet_data is not None and wavelet_distance is not None:
                self.process_segmentation(wavelet_data, wavelet_distance)
            segmentation_time = time.time() - segmentation_start_time
            self.segmentation_times.append(segmentation_time)
            
            # ============ ORIENTATION PROCESSING TIMING ============
            orientation_start_time = time.time()
            if getattr(self, "show_angle", False):
                current_orientation_data = self.get_current_orientation_data()
                if current_orientation_data is not None:
                    self.process_orientation_calculations(current_orientation_data)
            orientation_time = time.time() - orientation_start_time
            self.orientation_times.append(orientation_time)
            
            # Update timing statistics
            iteration_time = time.time() - iteration_start_time
            self.process_times.append(iteration_time)
            
            # Calculate refresh rate
            current_time = time.time()
            refresh_rate = 1.0 / (current_time - self.last_refresh_time)
            self.refresh_rates.append(refresh_rate)
            self.last_refresh_time = current_time
            
            self.iteration_count += 1
            
            # Print real-time statistics every 10 iterations
            if self.iteration_count % 10 == 0:
                avg_refresh_rate = np.mean(self.refresh_rates[-10:])
                avg_process_time = np.mean(self.process_times[-10:]) * 1000  # Convert to ms
                
                # Calculate average timing for each section (last 10 iterations)
                avg_acquisition_time = np.mean(self.acquisition_times[-10:]) * 1000
                avg_wavelet_time = np.mean(self.wavelet_times[-10:]) * 1000
                avg_segmentation_time = np.mean(self.segmentation_times[-10:]) * 1000
                avg_orientation_time = np.mean(self.orientation_times[-10:]) * 1000
                
                print(f"Iteration {self.iteration_count:4d} | "
                      f"Rate: {avg_refresh_rate:5.1f} Hz | "
                      f"Total: {avg_process_time:5.1f} ms | "
                      f"Acq: {avg_acquisition_time:4.1f} ms | "
                      f"Wav: {avg_wavelet_time:4.1f} ms | "
                      f"Seg: {avg_segmentation_time:4.1f} ms | "
                      f"Ori: {avg_orientation_time:4.1f} ms | "
                      f"SegActive: {self.current_segmentation_active}")
                
        except Exception as e:
            print(f"Error in process_iteration: {e}")

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

    def process_segmentation(self, wavelet_data, wavelet_distance):
        """Process segmentation using threshold-based approach - backend version without GUI"""
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
            
            self.current_segmentation_intensity = min(1.0, self.current_segmentation_intensity/3000)
            
            # Skip GUI updates (update_highlighted_regions_wavelet) for backend processing
            # This is where the original mixin would update plot_widget, but we skip it for performance
            
        except Exception as e:
            print(f"Error in segmentation: {e}")
            self.current_segmentation_active = False
            self.current_segmentation_intensity = 0.0

    def process_orientation_calculations(self, orientation_data):
        """Process orientation calculations without visualization"""
        try:
            relative_pitch, relative_roll, relative_yaw = orientation_data
            
            # Check for sudden changes in pitch or yaw (possible IMU glitch)
            if abs(relative_pitch - self.pitch_angles[-1]) > 45 or abs(relative_yaw - self.yaw_angles[-1]) > 45:
                # Try to get a new stable reference
                stable_quat = self.get_valid_quaternion()
                if stable_quat is not None:
                    self.initial_quat = stable_quat
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
            
            # Process image updates (simulated for timing)
            if self.segmentation_enabled:
                # Get coordinates for the tip position
                ix, iy = self.plot_coords_to_image_indices(-y, -x)
                
                # Simulate image region update (using mixin method)
                self.update_image_region(ix, iy, self.current_segmentation_intensity)
            
        except Exception as e:
            print(f"Error processing orientation calculations: {e}")

    def print_final_statistics(self):
        """Print final performance statistics"""
        print("\n" + "=" * 70)
        print("FINAL PERFORMANCE STATISTICS")
        print("=" * 70)
        
        if self.refresh_rates:
            print(f"Total iterations: {self.iteration_count}")
            print(f"Total runtime: {time.time() - self.start_time:.2f} seconds")
            print(f"Average refresh rate: {np.mean(self.refresh_rates):.2f} Hz")
            print(f"Max refresh rate: {np.max(self.refresh_rates):.2f} Hz")
            print(f"Min refresh rate: {np.min(self.refresh_rates):.2f} Hz")
            print(f"Refresh rate std dev: {np.std(self.refresh_rates):.2f} Hz")
            
        if self.process_times:
            avg_process_time = np.mean(self.process_times) * 1000
            print(f"\nTotal Process Time:")
            print(f"  Average: {avg_process_time:.2f} ms")
            print(f"  Max: {np.max(self.process_times) * 1000:.2f} ms")
            print(f"  Min: {np.min(self.process_times) * 1000:.2f} ms")
            print(f"  Std dev: {np.std(self.process_times) * 1000:.2f} ms")
            
        # Detailed timing breakdown for each processing section
        print(f"\nDetailed Timing Breakdown:")
        print(f"{'Section':<15} {'Enabled':<8} {'Avg (ms)':<10} {'Max (ms)':<10} {'Min (ms)':<10} {'Std (ms)':<10} {'% of Total':<10}")
        print("-" * 70)
        
        total_avg_time = np.mean(self.process_times) * 1000 if self.process_times else 1
        
        # Data Acquisition
        if self.acquisition_times:
            acq_avg = np.mean(self.acquisition_times) * 1000
            acq_max = np.max(self.acquisition_times) * 1000
            acq_min = np.min(self.acquisition_times) * 1000
            acq_std = np.std(self.acquisition_times) * 1000
            acq_pct = (acq_avg / total_avg_time) * 100
            bulk_mode = "Bulk" if self.use_bulk_acquisition else "Single"
            print(f"{'Acquisition':<15} {bulk_mode:<8} {acq_avg:<10.2f} {acq_max:<10.2f} {acq_min:<10.2f} {acq_std:<10.2f} {acq_pct:<10.1f}%")
        
        # Wavelet Processing
        if self.wavelet_times:
            wav_avg = np.mean(self.wavelet_times) * 1000
            wav_max = np.max(self.wavelet_times) * 1000
            wav_min = np.min(self.wavelet_times) * 1000
            wav_std = np.std(self.wavelet_times) * 1000
            wav_pct = (wav_avg / total_avg_time) * 100
            wav_enabled = "Yes" if self.enable_wavelet_processing else "No"
            print(f"{'Wavelet':<15} {wav_enabled:<8} {wav_avg:<10.2f} {wav_max:<10.2f} {wav_min:<10.2f} {wav_std:<10.2f} {wav_pct:<10.1f}%")
        
        # Segmentation Processing
        if self.segmentation_times:
            seg_avg = np.mean(self.segmentation_times) * 1000
            seg_max = np.max(self.segmentation_times) * 1000
            seg_min = np.min(self.segmentation_times) * 1000
            seg_std = np.std(self.segmentation_times) * 1000
            seg_pct = (seg_avg / total_avg_time) * 100
            seg_enabled = "Yes" if self.segmentation_enabled else "No"
            print(f"{'Segmentation':<15} {seg_enabled:<8} {seg_avg:<10.2f} {seg_max:<10.2f} {seg_min:<10.2f} {seg_std:<10.2f} {seg_pct:<10.1f}%")
        
        # Orientation Processing
        if self.orientation_times:
            ori_avg = np.mean(self.orientation_times) * 1000
            ori_max = np.max(self.orientation_times) * 1000
            ori_min = np.min(self.orientation_times) * 1000
            ori_std = np.std(self.orientation_times) * 1000
            ori_pct = (ori_avg / total_avg_time) * 100
            ori_enabled = "Yes" if self.show_angle else "No"
            print(f"{'Orientation':<15} {ori_enabled:<8} {ori_avg:<10.2f} {ori_max:<10.2f} {ori_min:<10.2f} {ori_std:<10.2f} {ori_pct:<10.1f}%")
            
        print("=" * 70)

    def run(self):
        """Run the benchmark loop"""
        print("Starting benchmark loop...")
        
        try:
            while self.running:
                self.process_iteration()
                
        except KeyboardInterrupt:
            print("\nBenchmark stopped by user")
        except Exception as e:
            print(f"Error in benchmark loop: {e}")
        finally:
            self.print_final_statistics()
            
            # Cleanup PicoScope
            try:
                self.status["stop"] = ps.ps3000aStop(self.chandle)
                assert_pico_ok(self.status["stop"])
                
                self.status["close"] = ps.ps3000aCloseUnit(self.chandle)
                assert_pico_ok(self.status["close"])
                print("PicoScope cleanup completed")
            except Exception as e:
                print(f"Error during cleanup: {e}")

if __name__ == '__main__':
    # You can configure the number of buffers here (default is 20)
    # For example: benchmark = BackendBenchmark(num_buffers=10) for 10 buffers
    # Or: benchmark = BackendBenchmark(num_buffers=40) for 40 buffers
    benchmark = BackendBenchmark(num_buffers=40)
    benchmark.run() 