from scipy.signal import butter, filtfilt

def butter_lowpass_filter(data, cutoff, fs=125e6, order=5):
    """
    Apply a zero-phase Butterworth lowpass filter.
    Default fs = 125,000,000 Hz (Red Pitaya native rate)
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def butter_bandpass_filter(data, lowcut, highcut, fs=125e6, order=5):
    """
    Apply a zero-phase Butterworth bandpass filter.
    Default fs = 125,000,000 Hz (Red Pitaya native rate)
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band', analog=False)
    return filtfilt(b, a, data)
