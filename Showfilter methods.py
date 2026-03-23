# import numpy as np
# import scipy.signal as signal
# import matplotlib.pyplot as plt
#
# # Generate example data
# t = np.linspace(0, 1, 1000)  # Time vector
# freq = 5  # Frequency of the sine wave
# data = np.sin(2 * np.pi * freq * t) + 0.5 * np.random.randn(1000)  # Add some random noise
#
# # Low-Pass Filtering
# cutoff_freq = 10  # Adjust as needed
# nyquist_freq = 0.5 * len(t)
# normal_cutoff = cutoff_freq / nyquist_freq
# b, a = signal.butter(4, normal_cutoff, btype='low', analog=False)
# low_pass_filtered = signal.filtfilt(b, a, data)
#
# # High-Pass Filtering
# cutoff_freq = 10  # Adjust as needed
# normal_cutoff = cutoff_freq / nyquist_freq
# b, a = signal.butter(4, normal_cutoff, btype='high', analog=False)
# high_pass_filtered = signal.filtfilt(b, a, data)
#
# # Band-Pass Filtering
# lowcut = 5
# highcut = 15
# lowcut_normal = lowcut / nyquist_freq
# highcut_normal = highcut / nyquist_freq
# b, a = signal.butter(4, [lowcut_normal, highcut_normal], btype='band', analog=False)
# band_pass_filtered = signal.filtfilt(b, a, data)
#
# # Median Filtering
# window_size = 21  # Adjust as needed
# median_filtered = signal.medfilt(data, kernel_size=window_size)
#
# # Plot original and filtered signals
#
# plt.figure(figsize=(12, 6))
# plt.subplot(4, 1, 1)
# plt.plot(t, data)
# plt.title('(a) Original Signal',fontsize=16)
# plt.legend()
#
# plt.subplot(4, 1, 2)
# plt.plot(t, low_pass_filtered)
# plt.title('(b) Low-Pass Filtered',fontsize=16)
# plt.legend()
#
# plt.subplot(4, 1, 3)
# plt.plot(t, high_pass_filtered)
# plt.title('(c) High-Pass Filtered',fontsize=16)
# plt.legend()
#
# plt.subplot(4, 1, 4)
# plt.plot(t, band_pass_filtered)
# plt.title('(d) Band-Pass Filtered',fontsize=16)
# plt.legend()
#
# plt.tight_layout()
# plt.show()
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import pywt

# Generate example data
t = np.linspace(0, 1, 1000)
frequency = 5
signal_data = np.sin(2 * np.pi * frequency * t) + 0.5 * np.random.randn(1000)

# Perform FFT
fft_result = np.fft.fft(signal_data)
freq_fft = np.fft.fftfreq(len(signal_data))

# Perform STFT
frequencies, times, stft_result = signal.stft(signal_data, fs=1000, nperseg=100, noverlap=50)

# Perform Wavelet Transform
wavelet_name = 'morl'
coeffs, freqs = pywt.cwt(signal_data, np.arange(1, 51), wavelet_name, sampling_period=1/1000)

# Create a figure with subplots
plt.figure(figsize=(12, 8))

# Plot original signal
plt.subplot(3, 1, 1)
plt.plot(t, signal_data, label='Original Signal')
plt.title('Original Signal')

# Plot FFT result
plt.subplot(3, 1, 2)
plt.plot(freq_fft, np.abs(fft_result), label='FFT Result')
plt.title('FFT')

# Plot STFT result
plt.subplot(3, 1, 3)
plt.pcolormesh(times, frequencies, np.abs(stft_result), shading='auto')
plt.colorbar(label='Magnitude')
plt.title('STFT')

# Adjust subplots layout
plt.tight_layout()

# Show the figure
plt.show()


plt.tight_layout()
plt.show()
