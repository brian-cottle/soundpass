import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the data
print("Loading data.csv...")
df = pd.read_csv('data.csv')

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Method 1: The "Waterfall" Heatmap (Best for visualizing 80 pulses together)
plt.figure(figsize=(14, 8))
# Transpose so time is on X-axis and pulse number is on Y-axis
im = plt.imshow(df.values.T, aspect='auto', cmap='viridis', origin='lower', 
               extent=[0, df.shape[0], 1, df.shape[1]])
plt.colorbar(im, label='Voltage (V)')
plt.title("Ultrasound Pulse Waterfall (80 Captures)")
plt.xlabel("Sample Index (8ns per sample)")
plt.ylabel("Pulse Number")
output_path1 = 'plots/ultrasound_waterfall.png'
plt.savefig(output_path1, dpi=300)
print(f"Waterfall plot saved to: {output_path1}")

# Method 2: Overlaying a few representative pulses to see the exact shape
plt.figure(figsize=(14, 6))
# Plot the 1st, 20th, 40th, 60th, and 80th pulse
pulses_to_plot = [df.columns[0], df.columns[19], df.columns[39], df.columns[59], df.columns[79]]
for i, col in enumerate(pulses_to_plot):
    plt.plot(df[col] + (i * 0.1), label=col, alpha=0.8, linewidth=1)

plt.title("Sample Pulses from the 80 Capture Run")
plt.xlabel("Sample Index (8ns per sample)")
plt.ylabel("Voltage (V) - Offset for visibility")
plt.legend()
plt.grid(True)
plt.tight_layout()
output_path2 = 'plots/ultrasound_samples.png'
plt.savefig(output_path2, dpi=300)
print(f"Sample line plot saved to: {output_path2}")

# Method 3: Averaged Waveform (The "Clean" Signal)
plt.figure(figsize=(14, 6))
averaged_pulse = df.mean(axis=1)
plt.plot(averaged_pulse, color='red', linewidth=1.5)
plt.title(f"Averaged Waveform (Mean of {df.shape[1]} Pulses)")
plt.xlabel("Sample Index (8ns per sample)")
plt.ylabel("Voltage (V)")
plt.grid(True)
plt.tight_layout()
output_path3 = 'plots/ultrasound_averaged.png'
plt.savefig(output_path3, dpi=300)
print(f"Averaged plot saved to: {output_path3}")