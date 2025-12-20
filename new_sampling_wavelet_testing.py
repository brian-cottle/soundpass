# %%
import pywt
import numpy as np
import pickle
from glob import glob
import os
import matplotlib.pyplot as plt
from live_orientation_utils import high_pass_filter, low_pass_filter
from natsort import natsorted
# %%

# example_dir = "/home/briancottle/Code/SoundPass/new_datasets/2025-05-30/1748622624_11"
example_dir = "/home/briancottle/Code/SoundPass/new_datasets/2025-09-15/1757955094_13"
sample_pkl_files = natsorted(glob(os.path.join(example_dir, "*.pkl")))

all_data = []
for sample_pkl_file in sample_pkl_files:
    with open(sample_pkl_file, "rb") as f:
        data = pickle.load(f)
        all_data.append(np.mean(data[0],axis=0))
# %%
all_data = np.array(all_data)
print(f'shape of loaded data: {all_data.shape}')



# %%

# Perform Multiresolution Analysis (MRA) on a single sample using Daubechies 4 wavelet with 8 levels
sample_idx = 434  # or any valid index
signal = all_data[sample_idx]
start_idx = 0



plt.figure(figsize=(12, 5))
plt.plot(
    np.arange(len(signal[start_idx:])), 
    signal[start_idx:], 
    color='royalblue', 
    linewidth=2, 
    label=f'Sample {sample_idx}'
)
plt.xlim(0, len(signal[start_idx:]))
plt.ylim(-500, 500)
plt.title(f"Waveform of Sample {sample_idx}", fontsize=16, fontweight='bold')
plt.xlabel("Sample Index", fontsize=14)
plt.ylabel("Amplitude", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
# %%

# Perform 8-level DWT using 'db4'
wavelet = 'db8'
level = 8
coeffs = pywt.mra(signal, wavelet, level=level, transform='dwt')
coeffs = np.array(coeffs)

start_idx = 1000
end_idx = 7000
print(f'start_idx: {start_idx}, end_idx: {end_idx}')
plt.figure(figsize=(10, 4))
plt.imshow(
    np.abs(coeffs[:,start_idx:end_idx]),
    aspect='auto',
    cmap='jet',
    origin='lower'
)
plt.title("Minimalistic Spectrogram from MRA Coefficients", fontsize=14)
plt.xlabel("Time (Sample Index)", fontsize=12)
plt.ylabel("MRA Level", fontsize=12)
plt.colorbar(label='Amplitude', shrink=0.8)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.imshow(
    np.abs(coeffs[:,start_idx:end_idx]),
    aspect='auto',
    cmap='jet',
    origin='lower'
)
plt.title("Minimalistic Spectrogram from MRA Coefficients", fontsize=14)
plt.xlabel("Time (Sample Index)", fontsize=12)
plt.ylabel("MRA Level", fontsize=12)
plt.colorbar(label='Amplitude', shrink=0.8)
plt.tight_layout()
plt.show()

start_idx = 0
end_idx = 30000


plt.figure(figsize=(10, 4))
plt.imshow(
    np.abs(coeffs[:,start_idx:end_idx]),
    aspect='auto',
    cmap='jet',
    origin='lower'
)
plt.title("Minimalistic Spectrogram from MRA Coefficients", fontsize=14)
plt.xlabel("Time (Sample Index)", fontsize=12)
plt.ylabel("MRA Level", fontsize=12)
plt.colorbar(label='Amplitude', shrink=0.8)
plt.tight_layout()
plt.show()
# %%

# Apply MRA-based denoising and extraction using the same approach as above

# Use the same wavelet and level as in the previous MRA example
wavelet = 'db4'
level = 4

# Perform MRA (multiresolution analysis) to get the detail signals at each level
mra_coeffs = pywt.mra(signal, wavelet, level=level, transform='dwt')
mra_coeffs = np.array(mra_coeffs)

# Take the absolute value of the signal
abs_mra_coeffs = np.abs(mra_coeffs)

median_coeff_value = np.median(abs_mra_coeffs)
threshold = median_coeff_value * 2
print(f'median coefficient value: {median_coeff_value}')
print(f'threshold: {threshold}')

thresholded_coeffs = abs_mra_coeffs.copy()
thresholded_coeffs[abs_mra_coeffs < threshold] = 0

reconstructed_signal = np.sum(thresholded_coeffs, axis=0)
original_signal = np.sum(abs_mra_coeffs, axis=0)

# next steps use a moving histogram approach to getting the threshold, sort of a local median thing within a window


plt.plot(original_signal)
plt.plot(reconstructed_signal)
plt.ylim(0, 20000)
# %%

low_pass_filtered_signal = low_pass_filter(abs(reconstructed_signal), cutoff=50, fs=1000, order=5)
high_pass_filtered_signal = high_pass_filter(low_pass_filtered_signal, cutoff=30, fs=1000, order=5)
abs_high_pass_filtered_signal = abs(high_pass_filtered_signal)
plt.plot(abs_high_pass_filtered_signal)
plt.ylim(-0, 400)
plt.show()


# %%


high_pass_filtered_signal = high_pass_filter(abs(signal), cutoff=20, fs=1000, order=5)
plt.plot(high_pass_filtered_signal)
plt.ylim(-20000, 20000)
plt.show()


# %%

plt.plot(signal)
plt.ylim(-200, 20000)
plt.show()


# %%

# Apply low-pass filter to the data
filter_data = low_pass_filter(signal, cutoff=20, fs=1000, order=5)

# Take absolute value of filtered data
abs_data = np.abs(filter_data)

# Wavelet decomposition
wavelet = 'db4'
coeffs = pywt.wavedec(filter_data, wavelet, level=6)

# Apply thresholding to detail coefficients (skip the first, which is the approximation)
threshold_value = 0.04 * np.max([np.max(np.abs(c)) for c in coeffs[1:]])
thresholded_coeffs = [coeffs[0]] + [pywt.threshold(c, threshold_value, mode='soft') for c in coeffs[1:]]

# Reconstruct the wavelet data
wavelet_data = pywt.waverec(thresholded_coeffs, wavelet)

# Calculate extracted wavelet data
extracted_wavelet_data = np.abs(abs_data - wavelet_data)
            
plt.plot(extracted_wavelet_data)
plt.ylim(-0, 400)
plt.show()

# %%

# MRA equivalent of the thresholding approach above
# Apply low-pass filter to the data (same as above)
filter_data = low_pass_filter(signal, cutoff=20, fs=500, order=5)

# Take absolute value of filtered data
abs_data = np.abs(filter_data)

# Perform MRA (multiresolution analysis) to get the detail signals at each level
wavelet = 'db4'
level = 2
mra_coeffs = pywt.mra(filter_data, wavelet, level=level, transform='dwt')
mra_coeffs = np.array(mra_coeffs)

# Apply thresholding to detail coefficients (skip the first, which is the approximation)
# Calculate threshold value similar to the traditional approach
# threshold_value = 0.04 * np.max([np.max(np.abs(c)) for c in mra_coeffs[1:]])
# thresholded_mra_coeffs = mra_coeffs.copy()
# for i in range(1, mra_coeffs.shape[0]):
#     thresholded_mra_coeffs[i] = pywt.threshold(mra_coeffs[i], threshold_value, mode='soft')

thresholded_mra_coeffs = mra_coeffs.copy()
thresholded_mra_coeffs[1:,:] = 0

# Reconstruct the wavelet data by summing the thresholded MRA components
wavelet_data_mra = np.sum(thresholded_mra_coeffs, axis=0)

# Calculate extracted wavelet data (same approach as above)
extracted_wavelet_data_mra = np.abs(abs_data - wavelet_data_mra)
using_filtered_data = abs_data - wavelet_data_mra
plt.plot(using_filtered_data)
plt.ylim(-0, 400)
plt.show()
# %%

# Visualize MRA coefficients before and after thresholding
start_idx = 1000
end_idx = 7000

# Plot original MRA coefficients
plt.figure(figsize=(12, 4))
plt.imshow(
    np.abs(mra_coeffs[:,start_idx:end_idx]),
    aspect='auto',
    cmap='jet',
    origin='lower'
)
plt.title("MRA Coefficients (Before Thresholding)", fontsize=14)
plt.xlabel("Time (Sample Index)", fontsize=12)
plt.ylabel("MRA Level", fontsize=12)
plt.colorbar(label='Amplitude', shrink=0.8)
plt.tight_layout()
plt.show()

# Plot thresholded MRA coefficients
plt.figure(figsize=(12, 4))
plt.imshow(
    np.abs(thresholded_mra_coeffs[:,start_idx:end_idx]),
    aspect='auto',
    cmap='jet',
    origin='lower'
)
plt.title("MRA Coefficients (After Thresholding)", fontsize=14)
plt.xlabel("Time (Sample Index)", fontsize=12)
plt.ylabel("MRA Level", fontsize=12)
plt.colorbar(label='Amplitude', shrink=0.8)
plt.tight_layout()
plt.show()

# Plot full signal range for better overview
start_idx_full = 0
end_idx_full = 30000

plt.figure(figsize=(12, 4))
plt.imshow(
    np.abs(mra_coeffs[:,start_idx_full:end_idx_full]),
    aspect='auto',
    cmap='jet',
    origin='lower'
)
plt.title("MRA Coefficients - Full Signal Range", fontsize=14)
plt.xlabel("Time (Sample Index)", fontsize=12)
plt.ylabel("MRA Level", fontsize=12)
plt.colorbar(label='Amplitude', shrink=0.8)
plt.tight_layout()
plt.show()

# %%

