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

# %% Proper wavelet sampling method


# Use the same wavelet and level as in the previous MRA example
wavelet = 'db4'
level = 5

# Perform MRA (multiresolution analysis) to get the detail signals at each level
mra_coeffs = pywt.mra(signal, wavelet, level=level, transform='dwt')
mra_coeffs = np.array(mra_coeffs)
mra_coeffs[0:2,:] = 0
# Take the absolute value of the signal
abs_mra_coeffs = np.abs(mra_coeffs)

original_signal = np.sum(abs_mra_coeffs, axis=0)
low_passed_original_signal = low_pass_filter(original_signal, cutoff=15, fs=1000, order=5)



# next steps use a moving histogram approach to getting the threshold, sort of a local median thing within a window
plt.plot(original_signal)
plt.plot(low_passed_original_signal)
plt.plot(high_passed_original_signal)
plt.ylim(0, 350)

# %%