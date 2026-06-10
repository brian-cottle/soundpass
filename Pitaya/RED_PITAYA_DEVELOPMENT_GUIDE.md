# Red Pitaya STEMlab 125-14 Development Guide

This guide serves as the technical handoff for the "SoundPass" ultrasound project. It contains the necessary commands, compiler flags, and architectural decisions required to continue development on the Red Pitaya.

## 1. Connection & Connectivity
*   **Hostname:** `rp-f0f296.local`
*   **Username:** `root`
*   **Password:** `root`
*   **Connection Method:** Direct Ethernet to host machine (Link-local).
*   **SSH Command:** `ssh root@rp-f0f296.local`
*   **Mac Target IP:** e.g., `169.254.86.152` or `169.254.244.170` (Note: This can change dynamically if the Mac sleeps or the cable is unplugged. Verify using: `ifconfig en0 | grep inet`).
*   **Filesystem State:** The OS is often Read-Only by default. To enable writing (e.g., to create directories or save logs), run: `mount -o remount,rw /`

## 2. Compilation & C API
The Red Pitaya hardware libraries are located in `/opt/redpitaya`. Compiling requires explicit paths for headers and shared objects.

### Required Header Includes (C)
```c
#include "rp.h"
#include "rp_hw-calib.h" // Required for rp_CalibrationReset
```

### The "Pro" Compile Command (Local Example)
This command links the necessary libraries and "hard-codes" the library path into the executable so `LD_LIBRARY_PATH` is not needed at runtime.
```bash
gcc -I/opt/redpitaya/include \
    -L/opt/redpitaya/lib \
    -Wl,-rpath,/opt/redpitaya/lib \
    YOUR_FILE.c -o YOUR_EXECUTABLE \
    -lrp -lrp-hw-calib
```

### Critical API Initialization
Always call `rp_CalibrationReset(false, false)` after `rp_Init()` to bypass potential factory calibration corruption (divide-by-zero errors).

## 3. High-Speed Data Workflow (AXI DMA)
To bypass the ~100Hz software bottleneck of the standard API, use the AXI DMA engine to stream data directly into reserved system RAM.

*   **RAM Limit:** 2MB (exactly 1,048,576 samples of `int16_t`).
*   **Sampling Rate:** Locked at 125 MS/s (8ns per sample) at `RP_DEC_1`.
*   **Strategy:** Capture a continuous block via AXI DMA, then process and slice the pulses directly in C on the Red Pitaya. The pulses are averaged in real-time to reduce noise, mean-centered to remove DC offset, and transmitted over TCP to the host machine.

## 4. Operational Development Loop (Mac <-> Red Pitaya)
All commands should be run from the local Mac terminal in the `Pitaya` directory.

### Push Code to RP:
To deploy the integrated IMU and AXI streaming code:
```bash
scp stream_axi_imu.c bno055.c bno055.h root@rp-f0f296.local:~/
```

### Compile on RP (once SSHed in):
```bash
gcc -I/opt/redpitaya/include -L/opt/redpitaya/lib -Wl,-rpath,/opt/redpitaya/lib stream_axi_imu.c bno055.c -o stream_axi_imu -lrp -lrp-hw-calib -lpthread -lm
```
*(Tip: Combine transfer and compilation into a single one-liner using `&&` to speed up the loop.)*

### Pull Data from RP (Legacy/Debug):
```bash
scp root@rp-f0f296.local:~/data_raw.bin ./
```

## 5. Running the System
The system is started in two parts. The Python visualizer must be started first so it can open the listening socket on port 5005.

### Step A: Start the Receiver (Mac)
```bash
python3 ./live_scope.py
```
*(Optionally pass `--no-imu` if running the non-IMU `stream_axi_mt` executable).*

### Step B: Start the Stream (Red Pitaya)
Execute the compiled binary via SSH, passing the Mac's Link-Local IP, trigger threshold (in ADC counts), optionally a custom pulse length, and optionally the number of 8ms buffers to average per frame (default: 1):
```bash
ssh root@rp-f0f296.local "./stream_axi_imu 169.254.86.152 20 6000 2"
```
*(Here, 20 is the raw ADC count threshold (~50mV), 6000 is the pulse length, and 2 is the number of 8ms buffers to average).*

## 6. Project Architecture
*   **Current State:** Live TCP Streaming of High-Speed Ultrasound + IMU Orientation.
*   **Data Capture & Processing (Red Pitaya):** C program (`stream_axi_imu.c`) performs continuous AXI DMA capture. It identifies ultrasound pulses in real-time, averages them, reads BNO055 IMU orientation and calibration status via I2C, and pushes this combined telemetry packet over a TCP socket (with MSG_NOSIGNAL) to the host Mac.
*   **Visualization (Host Mac):** PyQtGraph GUI (`live_scope.py`) runs a non-blocking TCP server, receives the telemetry stream, and live-plots the averaged pulse alongside a 3D polar representation of the IMU orientation.

## 7. TCP Telemetry Protocol
The Red Pitaya acts as the **TCP Client**, and the Mac Python GUI acts as the **TCP Server** (listening on Port 5005). The Pitaya detects dropped sockets and will automatically re-initialize the BNO055 and attempt to reconnect.

### Data Packet Structure
The payload is a continuous stream of 32-bit floats (`float32`). Each frame consists of:
1. `[0]`: **Number of Pulses** captured and averaged in this frame.
2. `[1-4]`: **IMU Quaternion** `[W, X, Y, Z]` directly from the BNO055.
3. `[5-8]`: **IMU Calibration Status** for `[System, Gyro, Accel, Mag]`. Each value is an integer `0-3` cast to float.
4. `[9+]`: **Averaged Pulse Data**. Length is dynamically defined by `pulse_len` (default 500 samples). This data has had its mean subtracted (DC offset removed) prior to transmission.

**Total Packet Bytes:** `(pulse_len + 9) * sizeof(float)`

## 8. Storage & Partitioning
The SD card partition may need expansion to utilize full capacity.
*   **Tool:** `/opt/redpitaya/sbin/resize.sh`
*   **Manual Grow:** `resize2fs /dev/mmcblk0p2` (after a reboot following `fdisk` resizing).

## 9. Development Best Practices & Safety
*   **Verification & Honesty**: Never assume system status or make unfounded claims about the codebase or hardware state. Every optimization, bug fix, or sensor reading must be positively verified via log outputs, return codes, or physical measurements.
*   **Step-by-Step Validation**: When modifying the C codebase or Python scripts:
    1. Confirm changes and their intended outcomes with the team before writing to files.
    2. Document and summarize the changes immediately after execution.
    3. Push and compile on the Red Pitaya using the target commands.
    4. Run manual or automated verification to prove the changes work as intended.
*   **Safety & Resetting**: If the IMU orientation or connection fails, verify the physical connection and check calibration status (`System, Gyro, Accel, Mag`). Do not proceed with uncalibrated sensor data if high precision is required.
