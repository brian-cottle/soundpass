import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Settings
TOTAL_SAMPLES = 1000000
PULSE_SIZE = 7500
NUM_PULSES = 80
TRIGGER_SENSITIVITY = 40 # Raw counts (~100mV in HV mode)

print("Loading raw binary data...")
try:
    # Read raw 16-bit integers
    big_buffer = np.fromfile('data_raw.bin', dtype=np.int16)
except FileNotFoundError:
    print("Error: data_raw.bin not found. Did you scp it from the Red Pitaya?")
    exit(1)

if len(big_buffer) != TOTAL_SAMPLES:
    print(f"Warning: Expected {TOTAL_SAMPLES} samples, got {len(big_buffer)}")

print("Extracting pulses...")
pulses = []
i = 1000 # Start after edge effects
baseline = big_buffer[100]

while i < (len(big_buffer) - PULSE_SIZE) and len(pulses) < NUM_PULSES:
    if big_buffer[i] > (baseline + TRIGGER_SENSITIVITY) or big_buffer[i] < (baseline - TRIGGER_SENSITIVITY):
        # Found a trigger!
        pulse = big_buffer[i - 500 : i - 500 + PULSE_SIZE]
        pulses.append(pulse)
        i += 10000 # Jump 80us ahead to skip the current pulse
        
        # Re-sample baseline for next pulse
        if i < len(big_buffer):
            baseline = big_buffer[i-100]
    else:
        i += 1

pulse_count = len(pulses)
print(f"Successfully extracted {pulse_count} pulses from the raw stream.")

if pulse_count > 0:
    print("Converting to Volts and plotting...")
    # Convert to Volts (HV mode: 8192 counts = 20V)
    # Transpose to get Shape: (PULSE_SIZE, pulse_count) to match CSV format
    pulse_array_volts = (np.column_stack(pulses).astype(np.float32) * 20.0) / 8192.0

    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join('plots', f"max_speed_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving plots to directory: {output_dir}")

    # 1. Waterfall
    plt.figure(figsize=(14, 8))
    im = plt.imshow(pulse_array_volts.T, aspect='auto', cmap='viridis', origin='lower', 
                   extent=[0, pulse_array_volts.shape[0], 1, pulse_array_volts.shape[1]])
    plt.colorbar(im, label='Voltage (V)')
    plt.title(f"Ultrasound Waterfall - Max Speed Capture ({timestamp})")
    plt.xlabel("Sample Index (8ns per sample)")
    plt.ylabel("Pulse Number")
    output_path1 = os.path.join(output_dir, 'ultrasound_waterfall_max.png')
    plt.savefig(output_path1, dpi=300)

    # 2. Sample Pulses
    plt.figure(figsize=(14, 6))
    indices = np.linspace(0, pulse_array_volts.shape[1]-1, min(5, pulse_array_volts.shape[1]), dtype=int)
    for i, idx in enumerate(indices):
        plt.plot(pulse_array_volts[:, idx] + (i * 0.1), label=f'Pulse_{idx+1}', alpha=0.8, linewidth=1)
    plt.title(f"Sample Pulses from Max Speed Run ({timestamp})")
    plt.xlabel("Sample Index (8ns per sample)")
    plt.ylabel("Voltage (V) - Offset for visibility")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    output_path2 = os.path.join(output_dir, 'ultrasound_samples_max.png')
    plt.savefig(output_path2, dpi=300)

    # 3. Averaged Waveform
    plt.figure(figsize=(14, 6))
    averaged_pulse = np.mean(pulse_array_volts, axis=1)
    plt.plot(averaged_pulse, color='red', linewidth=1.5)
    plt.title(f"Averaged Waveform - Mean of {pulse_array_volts.shape[1]} Pulses ({timestamp})")
    plt.xlabel("Sample Index (8ns per sample)")
    plt.ylabel("Voltage (V)")
    plt.grid(True)
    plt.tight_layout()
    output_path3 = os.path.join(output_dir, 'ultrasound_averaged_max.png')
    plt.savefig(output_path3, dpi=300)

    print("Done! Check your plots folder.")
else:
    print("No pulses found in the data stream. Signal might be smaller than the 100mV threshold.")
