import sys
import socket
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets
from scipy.spatial.transform import Rotation as R
from soundpass_utils import butter_bandpass_filter

import argparse

# --- CONFIGURATION ---
PORT = 5005

parser = argparse.ArgumentParser(description="SoundPass High-Speed Live Scope")
parser.add_argument("--pulse-len", type=int, default=6000, help="Number of samples per pulse")
parser.add_argument("--no-imu", action="store_true", help="Disable IMU data unpacking (for use with stream_axi_mt)")
args = parser.parse_args()

PULSE_LEN = args.pulse_len

# Packet format:
# With IMU:    [num_pulses, qw, qx, qy, qz, samples...] (PULSE_LEN + 5 floats)
# Without IMU: [num_pulses, samples...] (PULSE_LEN + 1 floats)
if args.no_imu:
    BUFFER_SIZE = (PULSE_LEN + 1) * 4
else:
    BUFFER_SIZE = (PULSE_LEN + 9) * 4

class LiveScope(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # --- DATA STATE ---
        self.avg_pulse = np.zeros(PULSE_LEN, dtype=np.float32)
        
        # --- UI SETUP ---
        self.setWindowTitle("SoundPass High-Speed Live Scope")
        self.resize(1200, 800)
        
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)

        # Plots
        self.win = pg.GraphicsLayoutWidget()
        layout.addWidget(self.win)
        
        # 1. Averaged Plot
        self.p1 = self.win.addPlot(title="Averaged Signal (Live)", col=0)
        self.p1.setLabel('left', 'Voltage', units='V')
        self.p1.setLabel('bottom', 'Distance', units='cm')
        self.p1.showGrid(x=True, y=True)
        self.curve1 = self.p1.plot(pen='c')
        self.p1.disableAutoRange()
        
        # 2. Polar Plot (Orientation)
        self.polar_plot = self.win.addPlot(title="Orientation", col=1)
        self.polar_plot.setAspectLocked(True)
        self.polar_plot.hideAxis('left')
        self.polar_plot.hideAxis('bottom')

        # Setup background image
        self.image_dimensions = 100
        self.bg_image_array = np.zeros((self.image_dimensions, self.image_dimensions), dtype=float)
        self.bg_image = pg.ImageItem(self.bg_image_array)
        self.bg_image.setRect(pg.QtCore.QRectF(-1, -1, 2, 2))
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
        
        # Crosshairs
        self.crosshair_v = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('y', width=2))
        self.crosshair_h = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('y', width=2))
        self.polar_plot.addItem(self.crosshair_v)
        self.polar_plot.addItem(self.crosshair_h)
        
        # Static center crosshairs
        static_pen = pg.mkPen(color=(150, 150, 150, 120), width=2, style=QtCore.Qt.PenStyle.DotLine)
        self.static_v = pg.InfiniteLine(angle=90, movable=False, pen=static_pen)
        self.static_h = pg.InfiniteLine(angle=0, movable=False, pen=static_pen)
        self.polar_plot.addItem(self.static_v)
        self.polar_plot.addItem(self.static_h)
        self.static_v.setPos(0)
        self.static_h.setPos(0)

        # Orientation State
        self.initial_quat = None
        self.pitch_angles = [0.0]
        self.roll_angles = [0.0]
        self.yaw_angles = [0.0]
        
        # Calculate X-axis (Distance in cm)
        # Red Pitaya 125MS/s = 8ns per sample. Speed of sound ~1540 m/s
        # Distance = (sample_idx * 8e-9 * 154000 cm/s) / 2
        self.x_data = np.arange(PULSE_LEN) * 0.000616

        # Control Panel
        self.controls = QtWidgets.QHBoxLayout()
        layout.addLayout(self.controls)
        
        self.thresh_label = QtWidgets.QLabel("Threshold (mV):")
        self.thresh_spin = QtWidgets.QSpinBox()
        self.thresh_spin.setRange(5, 2000)
        self.thresh_spin.setValue(50)
        
        self.holdoff_label = QtWidgets.QLabel("Holdoff (us):")
        self.holdoff_spin = QtWidgets.QSpinBox()
        self.holdoff_spin.setRange(5, 1000)
        self.holdoff_spin.setValue(20)

        self.y_range_label = QtWidgets.QLabel("Y Range (+/- V):")
        self.y_range_spin = QtWidgets.QDoubleSpinBox()
        self.y_range_spin.setRange(0.01, 1000.0)
        self.y_range_spin.setSingleStep(0.1)
        self.y_range_spin.setValue(0.3)

        self.x_range_label = QtWidgets.QLabel("X Range (pts):")
        self.x_range_spin = QtWidgets.QSpinBox()
        self.x_range_spin.setRange(10, 100000)
        self.x_range_spin.setSingleStep(100)
        self.x_range_spin.setValue(7000)

        self.fps_label = QtWidgets.QLabel("FPS: 0")
        
        # Explicit IMU text boxes
        self.yaw_label = QtWidgets.QLineEdit("Yaw: 0.0")
        self.yaw_label.setReadOnly(True)
        self.yaw_label.setStyleSheet("font-weight: bold; color: yellow; background-color: black; font-size: 14px; width: 100px;")
        
        self.pitch_label = QtWidgets.QLineEdit("Pitch: 0.0")
        self.pitch_label.setReadOnly(True)
        self.pitch_label.setStyleSheet("font-weight: bold; color: yellow; background-color: black; font-size: 14px; width: 100px;")
        
        self.roll_label = QtWidgets.QLineEdit("Roll: 0.0")
        self.roll_label.setReadOnly(True)
        self.roll_label.setStyleSheet("font-weight: bold; color: yellow; background-color: black; font-size: 14px; width: 100px;")

        self.controls.addWidget(self.thresh_label)
        self.controls.addWidget(self.thresh_spin)
        self.controls.addWidget(self.holdoff_label)
        self.controls.addWidget(self.holdoff_spin)
        self.controls.addWidget(self.y_range_label)
        self.controls.addWidget(self.y_range_spin)
        self.controls.addWidget(self.x_range_label)
        self.controls.addWidget(self.x_range_spin)
        self.controls.addStretch()
        self.controls.addWidget(self.yaw_label)
        self.controls.addWidget(self.pitch_label)
        self.controls.addWidget(self.roll_label)
        self.controls.addWidget(self.fps_label)
        
        # Second row of controls for Raw Quat
        self.controls2 = QtWidgets.QHBoxLayout()
        layout.addLayout(self.controls2)
        
        self.raw_w_label = QtWidgets.QLineEdit("W: 0")
        self.raw_w_label.setReadOnly(True)
        self.raw_w_label.setStyleSheet("font-weight: bold; color: cyan; background-color: black; font-size: 14px; width: 100px;")
        
        self.raw_x_label = QtWidgets.QLineEdit("X: 0")
        self.raw_x_label.setReadOnly(True)
        self.raw_x_label.setStyleSheet("font-weight: bold; color: cyan; background-color: black; font-size: 14px; width: 100px;")
        
        self.raw_y_label = QtWidgets.QLineEdit("Y: 0")
        self.raw_y_label.setReadOnly(True)
        self.raw_y_label.setStyleSheet("font-weight: bold; color: cyan; background-color: black; font-size: 14px; width: 100px;")
        
        self.raw_z_label = QtWidgets.QLineEdit("Z: 0")
        self.raw_z_label.setReadOnly(True)
        self.raw_z_label.setStyleSheet("font-weight: bold; color: cyan; background-color: black; font-size: 14px; width: 100px;")

        self.calib_label = QtWidgets.QLineEdit("Calib -> Sys: 0 | G: 0 | A: 0 | M: 0")
        self.calib_label.setReadOnly(True)
        self.calib_label.setStyleSheet("font-weight: bold; color: magenta; background-color: black; font-size: 14px; width: 280px;")

        self.controls2.addWidget(self.calib_label)
        self.controls2.addStretch()
        self.controls2.addWidget(self.raw_w_label)
        self.controls2.addWidget(self.raw_x_label)
        self.controls2.addWidget(self.raw_y_label)
        self.controls2.addWidget(self.raw_z_label)

        # Axis Selection Controls
        self.controls3 = QtWidgets.QHBoxLayout()
        layout.addLayout(self.controls3)
        
        self.x_axis_label = QtWidgets.QLabel("Polar X-Axis:")
        self.x_axis_combo = QtWidgets.QComboBox()
        self.x_axis_combo.addItems(['Roll', 'Pitch', 'Yaw'])
        self.x_axis_combo.setCurrentText('Pitch')
        
        self.y_axis_label = QtWidgets.QLabel("Polar Y-Axis:")
        self.y_axis_combo = QtWidgets.QComboBox()
        self.y_axis_combo.addItems(['Roll', 'Pitch', 'Yaw'])
        self.y_axis_combo.setCurrentText('Roll')

        self.filter_checkbox = QtWidgets.QCheckBox("Enable Filter (1-5 MHz)")
        self.filter_checkbox.setChecked(True)
        self.filter_checkbox.setStyleSheet("color: white; font-weight: bold; font-size: 13px;")

        self.controls3.addWidget(self.x_axis_label)
        self.controls3.addWidget(self.x_axis_combo)
        self.controls3.addWidget(self.y_axis_label)
        self.controls3.addWidget(self.y_axis_combo)
        self.controls3.addWidget(self.filter_checkbox)
        
        self.restart_btn = QtWidgets.QPushButton("Restart Stream")
        self.restart_btn.setStyleSheet("background-color: #8b0000; color: white; font-weight: bold; padding: 5px;")
        self.restart_btn.clicked.connect(self.restart_stream)
        self.controls3.addWidget(self.restart_btn)
        
        self.controls3.addStretch()

        # --- NETWORKING ---
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('0.0.0.0', PORT))
        self.server_socket.listen(1)
        self.server_socket.setblocking(False) # Non-blocking for the UI thread
        self.conn = None
        
        # --- TIMER FOR UPDATES ---
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(10) # 100Hz attempt

        import time
        self.time_module = time
        self.last_time = time.time()
        self.fps_history = []
        self.frame_count = 0
        print(f"Server listening on port {PORT}...")

    def restart_stream(self):
        print("Restart Stream clicked. Dropping connection to force reconnect...")
        if self.conn:
            self.conn.close()
            self.conn = None
            if hasattr(self, 'recv_buffer'):
                self.recv_buffer.clear()
                
        # Reset orientation baseline and metrics
        self.initial_quat = None
        self.frame_count = 0
        self.fps_history = []
        
        # Clear visual plots
        self.curve1.setData([], [])
        self.crosshair_v.setPos(0)
        self.crosshair_h.setPos(0)
        
        # Reset labels
        self.yaw_label.setText("Yaw: 0.0")
        self.pitch_label.setText("Pitch: 0.0")
        self.roll_label.setText("Roll: 0.0")
        self.raw_w_label.setText("W: 0")
        self.raw_x_label.setText("X: 0")
        self.raw_y_label.setText("Y: 0")
        self.raw_z_label.setText("Z: 0")

    def update(self):
        # 1. Check for connection
        if self.conn is None:
            try:
                self.conn, addr = self.server_socket.accept()
                self.conn.setblocking(False)
                self.pitaya_ip = addr[0]
                print(f"Connected by {addr}")
            except BlockingIOError:
                return
            except Exception as e:
                print(f"Accept error: {e}")
                return

        # 2. Try to read a full frame
        try:
            # We use a persistent buffer for partial reads
            if not hasattr(self, 'recv_buffer'):
                self.recv_buffer = bytearray()
            
            # Non-blocking read until buffer is full or would block
            while len(self.recv_buffer) < BUFFER_SIZE:
                try:
                    packet = self.conn.recv(BUFFER_SIZE - len(self.recv_buffer))
                    if not packet: 
                        print("Connection closed by peer.")
                        self.conn.close()
                        self.conn = None
                        self.recv_buffer = bytearray()
                        self.fps_history = []
                        self.frame_count = 0
                        return
                    self.recv_buffer.extend(packet)
                except BlockingIOError:
                    return # Wait for next timer tick for more data

            t_start = self.time_module.time()
            raw_frame = np.frombuffer(self.recv_buffer, dtype=np.float32).copy() # Clone to separate from bytearray
            self.recv_buffer = bytearray() # Clear for next frame
            t_net = self.time_module.time() - t_start
            
            num_pulses = int(raw_frame[0])
            
            if args.no_imu:
                avg_pulse = raw_frame[1:]
                self.imu_label.hide()
            else:
                qw, qx, qy, qz = raw_frame[1:5]
                sys_cal, gyro_cal, accel_cal, mag_cal = [int(v) for v in raw_frame[5:9]]
                avg_pulse = raw_frame[9:]
                
                self.calib_label.setText(f"Calib -> Sys: {sys_cal} | G: {gyro_cal} | A: {accel_cal} | M: {mag_cal}")
                
                # Stability Logic
                # BUG REPLICATION: To match Picoscope, we explicitly feed [w, x, y, z] to scipy's [x, y, z, w] input
                quat_array = np.array([qw, qx, qy, qz]) 
                norm = np.linalg.norm(quat_array)
                
                # Update raw labels with integer values
                self.raw_w_label.setText(f"W: {qw*16384:.0f}")
                self.raw_x_label.setText(f"X: {qx*16384:.0f}")
                self.raw_y_label.setText(f"Y: {qy*16384:.0f}")
                self.raw_z_label.setText(f"Z: {qz*16384:.0f}")
                
                # Reject torn frames (A valid quaternion must have a norm of ~1.0)
                if abs(norm - 1.0) < 0.1:
                    try:
                        current_quat = R.from_quat(quat_array)
                        if self.initial_quat is None:
                            self.initial_quat = current_quat
                        else:
                            relative_quat = current_quat * self.initial_quat.inv()
                            relative_euler = relative_quat.as_euler('xyz', degrees=True)
                            relative_roll = -relative_euler[0]
                            relative_yaw = relative_euler[1]
                            relative_pitch = relative_euler[2]
                            
                            self.pitch_angles.append(relative_pitch)
                            self.roll_angles.append(relative_roll)
                            self.yaw_angles.append(relative_yaw)
                            if len(self.pitch_angles) > 10:
                                self.pitch_angles.pop(0)
                                self.roll_angles.pop(0)
                                self.yaw_angles.pop(0)
                            
                            # Update labels
                            self.yaw_label.setText(f"Yaw: {relative_yaw:.1f}")
                            self.pitch_label.setText(f"Pitch: {relative_pitch:.1f}")
                            self.roll_label.setText(f"Roll: {relative_roll:.1f}")
                            
                            # Tip calculation and update crosshairs
                            d = 2
                            angles_dict = {'Roll': relative_roll, 'Pitch': relative_pitch, 'Yaw': relative_yaw}
                            x_angle = angles_dict[self.x_axis_combo.currentText()]
                            y_angle = angles_dict[self.y_axis_combo.currentText()]
                            
                            x_rad = np.radians(x_angle)
                            y_rad = np.radians(y_angle)
                            
                            y = d * np.sin(y_rad) * np.cos(x_rad)
                            x = d * np.sin(x_rad)
                            
                            self.crosshair_v.setPos(-x)
                            self.crosshair_h.setPos(-y)
                    except Exception as e:
                        print(f"Exception parsing quaternion: {e}")
                else:
                    print(f"REJECTED: Norm is {norm:.4f}, expected ~1.0")
            
            t_proc_start = self.time_module.time()
            t_proc = self.time_module.time() - t_proc_start
            
            t_plot_start = self.time_module.time()
            if num_pulses > 0:
                # Manual Y-axis range
                y_range = self.y_range_spin.value()
                self.p1.setYRange(-y_range, y_range, padding=0)

                # Manual X-axis range
                x_pts = self.x_range_spin.value()
                x_max = x_pts * 0.000616
                self.p1.setXRange(0, x_max, padding=0)

                # Apply filter if enabled
                plot_pulse = avg_pulse
                if self.filter_checkbox.isChecked():
                    try:
                        plot_pulse = butter_bandpass_filter(avg_pulse, lowcut=1e6, highcut=5e6, fs=125e6)
                    except Exception as fe:
                        print(f"Filter error: {fe}")

                # Update Plot
                self.curve1.setData(x=self.x_data, y=plot_pulse)
            t_plot = self.time_module.time() - t_plot_start

            # FPS Calculation
            now = self.time_module.time()
            dt = now - self.last_time
            self.last_time = now
            
            fps = 1.0 / dt if dt > 0.001 else 0.0
            if fps > 0:
                self.fps_history.append(fps)
                if len(self.fps_history) > 10:
                    self.fps_history.pop(0)
            
            # Use raw_frame processing outputs for display update
            avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0.0
            self.fps_label.setText(f"FPS: {avg_fps:.1f}")

            self.frame_count += 1
            if self.frame_count % 10 == 0:
                print(f"Avg FPS: {avg_fps:.1f} | Net: {t_net*1000:.1f}ms | Proc: {t_proc*1000:.1f}ms | Plot: {t_plot*1000:.1f}ms | Pulses: {num_pulses}")

        except Exception as e:
            print(f"Update error: {e}")
            self.conn = None

    def shutdown_pitaya(self):
        pitaya_ip = getattr(self, 'pitaya_ip', '169.254.135.224')
        print(f"Shutting down Pitaya C code at {pitaya_ip}...")
        expect_script = f"""
set timeout 2
spawn ssh -o StrictHostKeyChecking=no root@{pitaya_ip} "killall -9 stream_axi_imu"
expect {{
    "password:" {{
        send "root\\r"
        exp_continue
    }}
    eof
}}
"""
        import subprocess
        subprocess.run(['expect', '-c', expect_script])

    def closeEvent(self, event):
        self.shutdown_pitaya()
        if self.conn: self.conn.close()
        self.server_socket.close()

if __name__ == "__main__":
    import signal
    app = QtWidgets.QApplication(sys.argv)
    window = LiveScope()
    
    def sig_handler(*args):
        print("\nSignal received, shutting down...")
        window.close()
        
    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)
    
    window.show()
    sys.exit(app.exec_())
