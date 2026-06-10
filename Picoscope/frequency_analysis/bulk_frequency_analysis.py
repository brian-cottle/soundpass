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
# --- Configuration --- 
# Note: Ensure the volume/drive is mounted and accessible at this path. 
DATA_ROOT = "/Volumes/Samsung_T5/SoundPass_datasets/cadaver_annotated_datasets"
SAMPLING_INTERVAL = 4e-9  # 4 nanoseconds, corresponds to a 250 MHz sampling rate
IGNORE_THRESHOLD = 5000  # Ignore any segments or background starting before this sample
FFT_POINTS = 8192 # Number of points for the common frequency axis for interpolation

# %%
# --- Helper Functions --- 

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

def find_contiguous_segments(mask):
    """Finds contiguous blocks of 'True' in a boolean mask."""
    # Find indices where the mask value changes
    d = np.diff(mask.astype(int))
    # Find start indices (change from 0 to 1)
    start_indices = np.where(d == 1)[0] + 1
    # Find end indices (change from 1 to 0)
    end_indices = np.where(d == -1)[0] + 1

    # Handle cases where segment starts at the beginning
    if mask[0]:
        start_indices = np.insert(start_indices, 0, 0)
    # Handle cases where segment ends at the end
    if mask[-1]:
        end_indices = np.append(end_indices, len(mask))
    
    # Pair up start and end indices
    return list(zip(start_indices, end_indices))

# %%
# --- File Scanning --- 
print("Scanning for all .pkl files...")
all_files = sorted(glob.glob(f"{DATA_ROOT}/**/*.pkl", recursive=True))

print("Checking files for annotations...")
files_with_segmentation = []
for file_path in tqdm(all_files, desc="Scanning files"):
    try:
        with open(file_path, 'rb') as f:
            dataset = pickle.load(f)
            if 'ground_truth' in dataset and np.any(dataset['ground_truth'] > 0):
                files_with_segmentation.append(file_path)
    except Exception as e:
        print(f"\nCould not process file {file_path}: {e}")

print(f"\nScan complete. Found {len(files_with_segmentation)} files with annotations.")

# %%
# --- Main Analysis Loop --- 
all_annotated_powers = []
all_background_powers = []

# Define a common frequency axis for consistent interpolation
max_freq = 1 / (2 * SAMPLING_INTERVAL)
common_freq_axis = np.linspace(0, max_freq, FFT_POINTS)

for file_path in tqdm(files_with_segmentation, desc="Analyzing files"):
    try:
        with open(file_path, 'rb') as f:
            dataset = pickle.load(f)
        
        signal = dataset['data']
        ground_truth_mask = dataset['ground_truth'] > 0

        # 1. Find all contiguous annotated segments and filter out early ones
        contiguous_segments = find_contiguous_segments(ground_truth_mask)
        valid_annotated_signals = [
            signal[start:end] for start, end in contiguous_segments if start >= IGNORE_THRESHOLD
        ]

        if not valid_annotated_signals:
            continue

        # 2. Prepare the pool of valid background indices
        background_indices = np.where(~ground_truth_mask)[0]
        valid_background_pool = background_indices[background_indices >= IGNORE_THRESHOLD]
        
        # 3. For each valid annotated segment, find a background segment and analyze both
        for segment in valid_annotated_signals:
            seg_len = len(segment)
            if len(valid_background_pool) < seg_len:
                continue  # Not enough background data to sample from

            # Select a random start point for the background segment
            bg_start_index = np.random.choice(len(valid_background_pool) - seg_len)
            bg_indices = valid_background_pool[bg_start_index : bg_start_index + seg_len]
            background_segment = signal[bg_indices]
            
            # Analyze the annotated segment
            annotated_freqs, annotated_power = analyze_spectrum(segment, SAMPLING_INTERVAL)
            if annotated_freqs is not None and len(annotated_freqs) > 1:
                interp_annotated_power = np.interp(common_freq_axis, annotated_freqs, annotated_power)
                all_annotated_powers.append(interp_annotated_power)

            # Analyze the background segment
            background_freqs, background_power = analyze_spectrum(background_segment, SAMPLING_INTERVAL)
            if background_freqs is not None and len(background_freqs) > 1:
                interp_background_power = np.interp(common_freq_axis, background_freqs, background_power)
                all_background_powers.append(interp_background_power)

    except Exception as e:
        print(f"\nFailed to process {os.path.basename(file_path)}: {e}")


# %%
# --- Aggregate and Plot Final Results --- 
if not all_annotated_powers:
    print("No valid annotated segments found across all files after applying filters.")
else:
    # Calculate the average power spectra
    avg_annotated_power = np.mean(all_annotated_powers, axis=0)
    avg_background_power = np.mean(all_background_powers, axis=0) if all_background_powers else np.zeros_like(common_freq_axis)
    
    # Find the peak of the annotated spectrum to annotate the plot
    peak_index = np.argmax(avg_annotated_power)
    peak_freq_mhz = common_freq_axis[peak_index] / 1e6
    peak_power = avg_annotated_power[peak_index]

    # Plot the comparison
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(15, 7))

    ax.plot(common_freq_axis / 1e6, avg_annotated_power, label=f"Annotated Segments (Avg. of {len(all_annotated_powers)})", color='crimson', linewidth=1.5)
    if all_background_powers:
        ax.plot(common_freq_axis / 1e6, avg_background_power, label=f"Background Segments (Avg. of {len(all_background_powers)})", color='gray', alpha=0.8, linewidth=1)

    # Add a vertical line and annotation for the peak frequency
    ax.axvline(x=peak_freq_mhz, color='purple', linestyle='--', alpha=0.7)
    ax.annotate(f'Peak: {peak_freq_mhz:.2f} MHz',
                xy=(peak_freq_mhz, peak_power),
                xytext=(peak_freq_mhz + 10, peak_power * 0.5), # Adjust text position
                arrowprops=dict(facecolor='purple', shrink=0.05, alpha=0.7),
                fontsize=10,
                color='purple',
                horizontalalignment='left',
                verticalalignment='center')


    ax.set_title("Average Frequency Spectrum Comparison (Log Scale)")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Average Power (Log Scale)")
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.autoscale(enable=True, axis='x', tight=True)
    
    plt.tight_layout()
    plt.savefig("bulk_frequency_analysis.png")
    plt.show()

    print("\nAnalysis complete. Plot saved to 'bulk_frequency_analysis.png'.")

# %%