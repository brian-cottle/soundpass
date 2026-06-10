# Description of Averaged Ultrasound Data (.csv Files)

This document describes the format and characteristics of the averaged ultrasound signal data stored in `.csv` files.

## File Format

Each `.csv` file contains a single column of numerical values, representing an averaged ultrasound signal.

## Data Characteristics

-   **Data Points per File**: Each `.csv` file contains **7500** data points.
-   **Content**: The numerical values in each file are **averaged raw ADC (Analog-to-Digital Converter) counts**. These values are the result of averaging 80 consecutive A-mode signals recorded in rapid sequence, which are then averaged to create the 1D single data 
stream in each .csv.
-   **A-mode Sampling Interval**: **16 nanoseconds (ns)**
-   **Sampling Rate**: **62.5 MHz**

## Converting to Voltage

To convert the averaged ADC counts found in the `.csv` files to physical voltage values (in Volts), use the following formula:

`voltage = (averaged_adc_value / 32767) * 1.0`

Where:
-   `averaged_adc_value`: A value read directly from the `.csv` file.
-   `32767`: The maximum positive ADC value for the 16-bit digital acquisition system.
-   `1.0`: The input voltage range in Volts (representing the positive half of the +/- 1.0V range).

## Example: Loading and Converting Data

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
# Path to one of the CSV files
CSV_FILE_PATH = 'path/to/your/data_folder/1.csv'

# Signal characteristics for conversion
VOLTAGE_RANGE_V = 1.0
MAX_ADC_VALUE = 32767  # Maximum positive ADC value for the 16-bit system

# --- Processing ---
try:
    # Load the averaged ADC values from the CSV
    averaged_adc_signal = np.loadtxt(CSV_FILE_PATH, delimiter=",")

    # Convert the signal to Volts
    voltage_signal = (averaged_adc_signal / MAX_ADC_VALUE) * VOLTAGE_RANGE_V
    
    print(f"Successfully loaded and converted data from: {CSV_FILE_PATH}")
    print(f"Signal contains {len(voltage_signal)} data points.")
    print(f"Maximum voltage: {np.max(voltage_signal):.4f} V")
    print(f"Minimum voltage: {np.min(voltage_signal):.4f} V")

    # --- Optional: Plotting the Voltage Signal ---
    plt.figure(figsize=(12, 6))
    plt.plot(voltage_signal)
    plt.title("Averaged Ultrasound Signal (Volts)")
    plt.xlabel("Sample Index")
    plt.ylabel("Voltage (V)")
    plt.grid(True)
    plt.show()

except FileNotFoundError:
    print(f"Error: CSV file not found at {CSV_FILE_PATH}")
except Exception as e:
    print(f"An error occurred: {e}")

```