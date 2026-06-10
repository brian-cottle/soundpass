# %%
# --- Imports ---
import pickle
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy.signal import windows
from tqdm import tqdm

# %%
# --- Configuration and File Scanning ---
# Use glob to find all pickle files recursively.
# Note: Ensure the volume/drive is mounted and accessible at this path.
DATA_ROOT = "/Volumes/Samsung_T5/SoundPass_datasets/cadaver_annotated_datasets"
all_files = sorted(glob.glob(f"{DATA_ROOT}/**/*.pkl", recursive=True))

# %%
# --- Scan all files to find which have segmentations ---
print("Scanning all files to check for annotations...")
files_with_segmentation = []
files_without_segmentation = []

for file_path in tqdm(all_files, desc="Scanning files"):
    try:
        with open(file_path, 'rb') as f:
            dataset = pickle.load(f)
            if np.any(dataset['ground_truth'] > 0):
                files_with_segmentation.append(file_path)
            else:
                files_without_segmentation.append(file_path)
    except Exception as e:
        print(f"\nCould not process file {file_path}: {e}")

print(f"\nScan complete.")
print(f"Total files found: {len(all_files)}")
print(f"Files with annotations: {len(files_with_segmentation)}")
print(f"Files without annotations: {len(files_without_segmentation)}")


# %%
# --- Select File for Analysis ---
# For this analysis, we'll work with the list of files that have annotations.
analysis_list = files_with_segmentation

# --- USER: Choose which file to analyze by changing this index ---
file_index_to_analyze = 500  # Change this to select a different file from the 'analysis_list'
# ----------------------------------------------------------------

if not analysis_list or file_index_to_analyze >= len(analysis_list):
    raise FileNotFoundError(f"File index {file_index_to_analyze} is out of bounds or no files with segmentations were found.")

selected_file = analysis_list[file_index_to_analyze]
print(f"\nAnalyzing file: {selected_file}")

# Sampling parameters
SAMPLING_INTERVAL = 4e-9  # 4 nanoseconds, which corresponds to a 250 MHz sampling rate
IGNORE_THRESHOLD = 7000 # Samples before this index are excluded from analysis
# %% 
# --- Load Data ---
with open(selected_file, 'rb') as f:
    dataset = pickle.load(f)

signal = dataset['data']
ground_truth_mask = dataset['ground_truth']
time_axis = np.arange(len(signal)) * SAMPLING_INTERVAL

# Check if there are any annotated segments
has_segmentation = np.any(ground_truth_mask > 0)

if not has_segmentation:
    print("\nThis file contains no annotated segments.")
else:
    print("\nFile contains annotated segments. Proceeding with analysis.")

# %% 
# --- Plot Original Signal and Annotation ---
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(15, 5))

ax.plot(time_axis, signal, label="Signal", color='lightblue', linewidth=0.7)

ax.axvline(x=IGNORE_THRESHOLD * SAMPLING_INTERVAL, color='red', linestyle='--', label='Exclusion Threshold')

# Use a fill to highlight the annotated region
ax.fill_between(time_axis, ax.get_ylim()[0], ax.get_ylim()[1], where=ground_truth_mask > 0,
                facecolor='green', alpha=0.4, label="Annotated Segment")

ax.set_title(f"Signal and Annotation\n{os.path.basename(selected_file)}", fontsize=14)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.legend()
plt.tight_layout()
plt.show()

# %% 
# --- Frequency Analysis Function ---
def analyze_spectrum(segment, sampling_interval):
    """Computes the frequency spectrum of a signal segment."""
    if len(segment) == 0:
        return None, None

    # Apply a Hann window to reduce spectral leakage from segment edges
    window = windows.hann(len(segment))
    windowed_segment = segment * window

    # Compute FFT
    fft_result = np.fft.fft(windowed_segment)
    # Compute the power spectrum (magnitude squared)
    power_spectrum = np.abs(fft_result)**2

    # Compute the frequency bins for the FFT result
    frequencies = np.fft.fftfreq(len(segment), d=sampling_interval)

    # Keep only the positive frequencies for the plot
    positive_mask = frequencies >= 0
    return frequencies[positive_mask], power_spectrum[positive_mask]

# %% 
# --- Perform Analysis and Plot Spectra ---
if has_segmentation:
    # 1. Extract the annotated signal segment
    annotated_signal = signal[ground_truth_mask > 0]
    annotated_length = len(annotated_signal)
    print(f"Found annotated segment of length: {annotated_length} samples.")

    # 2. Extract a background segment for comparison
    # We'll take a random chunk of the background with the same length
    background_signal = np.array([])
    background_indices = np.where(ground_truth_mask == 0)[0]

    if len(background_indices) >= annotated_length:
        start_index = np.random.choice(len(background_indices) - annotated_length)
        background_segment_indices = background_indices[start_index : start_index + annotated_length]
        background_signal = signal[background_segment_indices]
        print(f"Using background segment of length: {len(background_signal)} samples.")
    else:
        # Fallback if background is shorter than annotation (unlikely but safe)
        background_signal = signal[ground_truth_mask == 0]
        print(f"Warning: Background is shorter than annotation. Using all {len(background_signal)} background samples.")


    # 3. Analyze both segments
    annotated_freqs, annotated_power = analyze_spectrum(annotated_signal, SAMPLING_INTERVAL)
    background_freqs, background_power = analyze_spectrum(background_signal, SAMPLING_INTERVAL)

    # 4. Plot the comparison
    fig, ax = plt.subplots(figsize=(15, 7))

    if annotated_freqs is not None:
        ax.plot(annotated_freqs / 1e6, annotated_power, label="Annotated Segment", color='crimson', linewidth=1)
    if background_freqs is not None:
        ax.plot(background_freqs / 1e6, background_power, label="Background", color='gray', alpha=0.7, linewidth=1)

    ax.set_title("Frequency Spectrum Comparison (Log Scale)")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Power (Log Scale)")
    ax.set_yscale('log')  # Log scale is often better for viewing power spectra
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

else:
    # If no annotations, just analyze the whole signal as a baseline
    print("Analyzing the spectrum of the entire signal as no annotations were found.")
    full_freqs, full_power = analyze_spectrum(signal, SAMPLING_INTERVAL)

    fig, ax = plt.subplots(figsize=(15, 7))
    if full_freqs is not None:
        ax.plot(full_freqs / 1e6, full_power, label="Full Signal Spectrum", color='dodgerblue')
        
    ax.set_title("Frequency Spectrum of Full Signal (Log Scale)")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Power (Log Scale)")
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

# %%