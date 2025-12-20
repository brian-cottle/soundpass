# %%
import os
import numpy as np
from matplotlib import pyplot as plt
from glob import glob
from natsort import natsorted
import pickle
from scipy.signal import butter, filtfilt
import pywt
from scipy.ndimage import gaussian_filter1d

def low_pass_filter(data, cutoff=10, fs=1000, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# %%

directory = '/home/briancottle/Code/SoundPass/new_datasets/2025-05-30/1748622624_11'

file_names = natsorted(glob(os.path.join(directory, "*.pkl")))
all_data_array = []
all_file_names = []
distances = []
for file_name in file_names:
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
        averaged_data = np.mean(data[0], axis=0)
        all_data_array.append(averaged_data)
        distances.append(data[1])
        all_file_names.append(file_name)
# %%

sample_index = 2600
data = all_data_array[sample_index][7000:]
distance = distances[sample_index][7000:]
import time

start_time = time.time()

filter_data = low_pass_filter(data, cutoff=20, fs=1000, order=5)

abs_data = np.abs(filter_data)
wavelet = 'db4'
coeffs = pywt.wavedec(filter_data, wavelet, level=4)
# Apply thresholding to detail coefficients (skip the first, which is the approximation)
threshold_value = 0.04 * np.max([np.max(np.abs(c)) for c in coeffs[1:]])  # Example threshold, can be tuned
thresholded_coeffs = [coeffs[0]] + [pywt.threshold(c, threshold_value, mode='soft') for c in coeffs[1:]]

wavelet_data = pywt.waverec(thresholded_coeffs, wavelet)
extracted_wavelet_data = np.abs(abs_data - wavelet_data)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.4f} seconds")

plt.figure(figsize=(16, 8))
# plt.plot(distance, data, label='Low-pass Filtered', color='red', alpha=0.5)
plt.plot(distance, filter_data, label='Raw Data', color='blue', alpha=0.2)
plt.plot(distance, extracted_wavelet_data, label='Mexican Hat Wavelet', color='green',alpha=0.6)
plt.legend()
plt.xlabel('Distance')
plt.ylabel('Amplitude')
plt.title('Signal Analysis with Mexican Hat Wavelet')

# %%

weighting_array = np.zeros(len(distance))
weighting_array[2500:17500] = 1
# Apply Gaussian filter to smooth the weighting array
sigma = 1000  # Standard deviation for the Gaussian filter
weighting_array = gaussian_filter1d(weighting_array, sigma=sigma)
plt.plot(weighting_array, label='Weighting Array')