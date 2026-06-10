import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

# Load the data
file_name = 'data_axi.csv'
print(f"Loading {file_name}...")
df = pd.read_csv(file_name)

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Get current timestamp for unique filenames
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create a unique subdirectory for this batch of plots
output_dir = os.path.join('plots', timestamp)
os.makedirs(output_dir, exist_ok=True)
print(f"Saving plots to directory: {output_dir}")

# Method 1: The "Waterfall" Heatmap
plt.figure(figsize=(14, 8))
im = plt.imshow(df.values.T, aspect='auto', cmap='viridis', origin='lower', 
               extent=[0, df.shape[0], 1, df.shape[1]])
plt.colorbar(im, label='Voltage (V)')
plt.title(f"Ultrasound Pulse Waterfall - AXI Capture ({timestamp})")
plt.xlabel("Sample Index (8ns per sample)")
plt.ylabel("Pulse Number")
output_path1 = os.path.join(output_dir, 'ultrasound_waterfall_axi.png')
plt.savefig(output_path1, dpi=300)
print(f"Waterfall plot saved to: {output_path1}")

# Method 2: Overlaying a few representative pulses
plt.figure(figsize=(14, 6))
# Plot 5 pulses evenly spaced
indices = np.linspace(0, df.shape[1]-1, 5, dtype=int)
pulses_to_plot = [df.columns[i] for i in indices]

for i, col in enumerate(pulses_to_plot):
    plt.plot(df[col] + (i * 0.1), label=col, alpha=0.8, linewidth=1)

plt.title(f"Sample Pulses from AXI Run ({timestamp})")
plt.xlabel("Sample Index (8ns per sample)")
plt.ylabel("Voltage (V) - Offset for visibility")
plt.legend()
plt.grid(True)
plt.tight_layout()
output_path2 = os.path.join(output_dir, 'ultrasound_samples_axi.png')
plt.savefig(output_path2, dpi=300)
print(f"Sample line plot saved to: {output_path2}")

# Method 3: Averaged Waveform
plt.figure(figsize=(14, 6))
averaged_pulse = df.mean(axis=1)
plt.plot(averaged_pulse, color='red', linewidth=1.5)
plt.title(f"Averaged Waveform - AXI Mean of {df.shape[1]} Pulses ({timestamp})")
plt.xlabel("Sample Index (8ns per sample)")
plt.ylabel("Voltage (V)")
plt.grid(True)
plt.tight_layout()
output_path3 = os.path.join(output_dir, 'ultrasound_averaged_axi.png')
plt.savefig(output_path3, dpi=300)
print(f"Averaged plot saved to: {output_path3}")
