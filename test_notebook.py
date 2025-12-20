# %%
import pickle
import matplotlib.pyplot as plt
import numpy as np
# %%

# Load a pickle file
with open('/home/briancottle/Code/SoundPass/new_datasets/2025-09-15/1757951824786_time0.pkl', 'rb') as file:
    data = pickle.load(file)

# %%
example_1 = data['data'][5000:10000]
x_1 = np.arange(len(example_1))
plt.plot(x_1,example_1)

# %%
steps = 30
example_2 = example_1[0::steps]
x_2 = x_1[0::steps]
plt.plot(example_2)

# %%


# Plot a subsection of data['data'] in a larger figure
subsection = np.mean(data[0],axis=0)
x_subsection = np.arange(len(subsection))/2

plt.figure(figsize=(16, 8))  # Create a larger figure
plt.plot(x_subsection, subsection)
plt.title("Subsection of data['data']")
plt.xlabel("Index")
plt.ylabel("Value")
plt.show()
# %%
averaged_data = np.mean(data[0], axis=0)
# Perform Fourier Transform on the averaged_data
sampling_frequency = 1 / (2e-9)  # Sampling frequency in Hz
fft_result = np.fft.fft(averaged_data)
frequencies = np.fft.fftfreq(len(averaged_data), d=1/sampling_frequency)

# Take the positive half of the spectrum
positive_frequencies = frequencies[:len(frequencies)//2]
positive_fft_result = np.abs(fft_result[:len(fft_result)//2])

# Plot the Fourier Transform
plt.figure(figsize=(16, 8))
plt.plot(positive_frequencies, positive_fft_result)
plt.title("Fourier Transform of Averaged Data")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
# Find the indices of the four highest peaks
# Find the index of the highest peak
peak_index = np.argmax(positive_fft_result)

# Annotate the highest peak in MHz
plt.annotate(f'{positive_frequencies[peak_index] / 1e6:.2f} MHz',
             (positive_frequencies[peak_index], positive_fft_result[peak_index]),
             textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10, color='red')

plt.grid()
plt.show()

# %%
