# %%
import pywt
import numpy as np
import pickle
from glob import glob
import os
import matplotlib.pyplot as plt
# %%

example_dir = "/home/briancottle/Code/SoundPass/new_datasets/2025-05-30/1748622624_11"
sample_pkl_files = glob(os.path.join(example_dir, "*.pkl"))

all_data = []
for sample_pkl_file in sample_pkl_files:
    with open(sample_pkl_file, "rb") as f:
        data = pickle.load(f)
        all_data.append(np.mean(data[0],axis=0))
# %%
all_data = np.array(all_data)




# %%

# Perform Multiresolution Analysis (MRA) on a single sample using Daubechies 4 wavelet with 8 levels
sample_idx = 450  # or any valid index
downsample_factor = 4
signal = all_data[sample_idx, ::downsample_factor]
start_idx = 7000//downsample_factor



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

start_idx = 17000//downsample_factor
end_idx = 20000//downsample_factor
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

start_idx = 0//downsample_factor
end_idx = 5000//downsample_factor


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
abs_signal = np.abs(signal)

# Threshold each MRA component (except the approximation, i.e., the first row)
# We'll use a similar thresholding strategy as before
threshold_value = 0.04 * np.max(np.abs(mra_coeffs[1:]))
thresholded_mra_coeffs = mra_coeffs.copy()
for i in range(1, mra_coeffs.shape[0]):
    thresholded_mra_coeffs[i] = pywt.threshold(mra_coeffs[i], threshold_value, mode='soft')

# Reconstruct the denoised signal by summing the thresholded MRA components
wavelet_denoised_mra = np.sum(thresholded_mra_coeffs, axis=0)

# Calculate extracted wavelet data
extracted_wavelet_data_mra = np.abs(abs_signal - wavelet_denoised_mra)

# Optionally, plot the results for visualization
plt.figure(figsize=(12, 6))
plt.plot(signal, label='Original Signal', alpha=0.5)
plt.plot(wavelet_denoised_mra, label='MRA Wavelet Denoised', alpha=0.7)
plt.plot(extracted_wavelet_data_mra, label='Extracted Wavelet Data (MRA)', alpha=0.7)
plt.legend()
plt.title("MRA Wavelet Denoising and Extraction")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.ylim(-500, 500)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.imshow(
    np.abs(mra_coeffs[:,4000:7000]),
    aspect='auto',
    cmap='jet',
    origin='lower'
)
# plt.clim(0, 150)

plt.title("Minimalistic Spectrogram from MRA Coefficients", fontsize=14)
plt.xlabel("Time (Sample Index)", fontsize=12)
plt.ylabel("MRA Level", fontsize=12)
plt.colorbar(label='Amplitude', shrink=0.8)
plt.tight_layout()
plt.show()


# %%

plt.figure(figsize=(10, 4))
plt.imshow(
    np.abs(thresholded_mra_coeffs[:,start_idx:end_idx]),
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

# Global threshold: keep top 5% amplitudes
threshold = np.percentile(np.abs(mra_coeffs[:,1500:-500]), 95)
mask = np.abs(mra_coeffs) > threshold
filtered_spec = mra_coeffs * mask
filtered_spec[0,:] = 0
filtered_spec[3:,:] = 0

reconstructed_signal = np.sum(filtered_spec, axis=0)

plt.figure(figsize=(10, 4))
plt.imshow(
    np.abs(filtered_spec[:,1500:-500]),
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

start_idx = 1500
end_idx = 7000

plt.figure(figsize=(10, 4))
plt.plot(np.abs(signal[start_idx:end_idx]), label='Original Signal', alpha=0.5)
plt.plot(np.abs(reconstructed_signal[start_idx:end_idx]), label='Reconstructed Signal', alpha=0.7)
plt.legend()
plt.title("Reconstructed Signal", fontsize=14)
plt.xlabel("Sample Index", fontsize=12)
plt.ylabel("Amplitude", fontsize=12)
plt.tight_layout()
plt.show()


# %%
