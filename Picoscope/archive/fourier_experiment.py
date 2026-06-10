# %%
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import glob
# %%
target_directory = '/home/briancottle/Code/SoundPass/2025-03-29/1743277613_open_air_pinch_no2'

# Find all pickle files in the target directory
pickle_files = glob.glob(f"{target_directory}/*.pkl")

# Load all pickle files into a list
all_data = []
for pickle_file in pickle_files:
    with open(pickle_file, 'rb') as file:
        data = pickle.load(file)
        averaged_data = np.mean(data[0], axis=0)
        all_data.append(averaged_data)


# Perform Fourier Transform on the averaged_data
sampling_frequency = 1 / (2e-9)  # Sampling frequency in Hz
# Average the FFT results
average_fft_result = np.mean([np.abs(np.fft.fft(data)[:len(data)//2]) for data in all_data], axis=0)
positive_fft_result = average_fft_result

frequencies = np.fft.fftfreq(len(averaged_data), d=1/sampling_frequency)

# Take the positive half of the spectrum
positive_frequencies = frequencies[:len(frequencies)//2]



# Plot the Fourier Transform
plt.figure(figsize=(16, 8))
plt.plot(positive_frequencies, positive_fft_result)
plt.title("Fourier Transform of Averaged Data: open air pinch")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
# Find the indices of the four highest peaks
# Find the index of the highest peak
# Find all peaks with a minimum distance of 10 indices between them
peaks, _ = find_peaks(positive_fft_result, distance=40)

# Get the top 5 peaks based on amplitude
top_peaks = sorted(peaks, key=lambda x: positive_fft_result[x], reverse=True)[:3]

# Annotate the top 5 peaks in MHz
for i, peak_index in enumerate(top_peaks):
    plt.annotate(f'{positive_frequencies[peak_index] / 1e6:.2f} MHz',
                 (positive_frequencies[peak_index], positive_fft_result[peak_index]),
                 textcoords="offset points", xytext=(0, -20 * (i + 1)), ha='center', fontsize=15, color='red')

plt.grid()
plt.show()

# %%
