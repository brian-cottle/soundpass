# Documentation for Saved Ultrasound Data Files

This document details the structure and contents of the `.pkl` files containing data from the ultrasound sensor system.

## Overview

Each `.pkl` file represents a single data capture event. It contains raw signal data from a digital oscilloscope, a corresponding distance vector, and orientation data from an inertial measurement unit (IMU). The data is serialized into a binary format using Python's `pickle` module.

## File Format

A single Python `list` object is stored in each `.pkl` file. This list contains three elements in the following order:

`[signal_data, distance_vector, orientation_data]`

## Data Structure Details

### 1. `signal_data`

- **Type**: `numpy.ndarray`
- **Shape**: `(80, 7500)`
- **Data Type**: `int16`
- **Description**: A 2D array containing the raw, unscaled 16-bit integer values from the digital oscilloscope.
  - The array consists of **80** consecutive measurements (buffers).
  - Each measurement contains **7500** samples.
  - These raw ADC (Analog-to-Digital Converter) values can be converted to voltage (in Volts) based on the acquisition's **+/- 1V** range and the maximum possible ADC value (typically 32,767 for a 16-bit ADC). The conversion formula is: `voltage = (adc_value / max_adc_value) * voltage_range_in_volts`.

### 2. `distance_vector`

- **Type**: `numpy.ndarray`
- **Shape**: `(7500,)`
- **Data Type**: `float64`
- **Description**: A 1D array corresponding to the samples in each signal buffer. Each value represents a calculated physical distance in **centimeters (cm)** from the sensor.

### 3. `orientation_data`

- **Type**: `list` of `float`
- **Shape**: `[3]`
- **Description**: A list containing the relative Euler angles from the BNO055 IMU sensor at the time of capture. The values represent the orientation of the sensor relative to its initial position. If orientation data was not available at the time of capture, this will be `[0, 0, 0]`.
  - **Format**: `[pitch, roll, yaw]`
  - **Units**: Degrees

## Signal Acquisition Characteristics

The data was captured using the following settings on the digital oscilloscope:

- **Sampling Interval**: **16 ns**
- **Sampling Rate**: **62.5 MHz**
- **Samples per Measurement (Buffer)**: **7500**
- **Measurements (Buffers) per File**: **80**
- **Total Samples per File**: 600,000 (80 buffers × 7500 samples)
- **Input Voltage Range**: **+/- 1.0 Volt**
- **Trigger Condition**: The acquisition is triggered on a rising edge of the input signal with a threshold of **500 mV**.

## Loading Example

The following Python script demonstrates how to load and interpret the data from a `.pkl` file.

```python
import pickle
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
# Replace with the actual path to your .pkl file
FILE_PATH = 'path/to/your/data_file.pkl'

# Oscilloscope settings known from the data acquisition process
VOLTAGE_RANGE_V = 1.0  # The captured voltage range was +/- 1.0V
MAX_ADC_VALUE = 32767  # Maximum ADC value for a 16-bit oscilloscope

def load_and_process_data(file_path):
    """Loads and processes the data from a single pickle file."""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        # Unpack the data from the list
        raw_signal_data, distance_cm, orientation_degrees = data

        # --- Data Inspection ---
        print(f"File loaded successfully: {file_path}")
        print("-" * 30)

        # Raw Signal Data
        print("Raw Signal Data (`signal_data`):")
        print(f"  - Type: {type(raw_signal_data)}")
        print(f"  - Shape: {raw_signal_data.shape}")
        print(f"  - Data Type: {raw_signal_data.dtype}")

        # Distance Vector
        print("\nDistance Vector (`distance_cm`):")
        print(f"  - Type: {type(distance_cm)}")
        print(f"  - Shape: {distance_cm.shape}")
        print(f"  - Data Type: {distance_cm.dtype}")

        # Orientation Data
        print("\nOrientation Data (`orientation_degrees`):")
        print(f"  - Type: {type(orientation_degrees)}")
        print(f"  - Values (pitch, roll, yaw): {orientation_degrees} degrees")
        print("-" * 30)

        # --- Data Processing ---
        # Average the 80 buffers to get a cleaner signal
        averaged_adc_signal = np.mean(raw_signal_data, axis=0)

        # Convert the averaged ADC values to voltage
        voltage_signal = (averaged_adc_signal / MAX_ADC_VALUE) * VOLTAGE_RANGE_V

        return distance_cm, voltage_signal

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None, None
    except Exception as e:
        print(f"An error occurred while loading or processing the file: {e}")
        return None, None

def plot_data(distance, voltage):
    """Plots the processed signal."""
    if distance is None or voltage is None:
        return
        
    plt.figure(figsize=(12, 6))
    plt.plot(distance, voltage)
    plt.title("Averaged Signal vs. Distance")
    plt.xlabel("Distance (cm)")
    plt.ylabel("Voltage (V)")
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    dist, volt = load_and_process_data(FILE_PATH)
    if dist is not None:
        plot_data(dist, volt)

```