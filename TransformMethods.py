# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import stft, chirp
# import pywt
#
# # Create a sample signal
# fs = 1000  # Sampling frequency
# t = np.linspace(0, 5, 5 * fs, endpoint=False)  # Time vector
# signal = chirp(t, f0=10, f1=100, t1=5, method='quadratic')  # Chirp signal
#
# # Compute FFT
# fft_freq = np.fft.fftshift(np.fft.fftfreq(len(signal), 1/fs))
# fft_amplitude = np.abs(np.fft.fftshift(np.fft.fft(signal)))
#
# # Compute STFT
# f, t, Zxx = stft(signal, fs=fs, nperseg=100, noverlap=50)
# Zxx = np.abs(Zxx)
#
# # Compute Wavelet Transform (using the Morlet wavelet)
# wavelet_transform, _ = pywt.cwt(signal, scales=np.arange(1, 50), wavelet='morl')
#
# # Create a figure with subplots
# fig, axs = plt.subplots(4, 1, figsize=(8, 10))
# axs[0].plot(signal, t)
# axs[0].set_title('Original')
# axs[0].set_xlabel('Frequency (Hz)')
# axs[0].set_ylabel('Amplitude')
# # Plot FFT
# axs[1].plot(fft_freq, fft_amplitude)
# axs[1].set_title('FFT')
# axs[1].set_xlabel('Frequency (Hz)')
# axs[1].set_ylabel('Amplitude')
#
# # Plot STFT
# axs[2].imshow(10 * np.log10(Zxx), extent=[t.min(), t.max(), f.min(), f.max()], aspect='auto', cmap='inferno')
# axs[2].set_title('STFT')
# axs[2].set_xlabel('Time (s)')
# axs[2].set_ylabel('Frequency (Hz)')
#
# # Plot Wavelet Transform
# im = axs[3].imshow(np.abs(wavelet_transform), extent=[t.min(), t.max(), 1, 50], aspect='auto', cmap='jet')
# axs[3].set_title('Wavelet Transform')
# axs[3].set_xlabel('Time (s)')
# axs[3].set_ylabel('Scale')
#
# # Add colorbars
# cbar1 = fig.colorbar(im, ax=axs[2])
# cbar1.set_label('Magnitude')
#
# plt.tight_layout()
# plt.show()
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import pywt

# Create a sample signal
fs = 1000  # Sampling frequency (Hz)
t1 = np.linspace(0, 1, fs, endpoint=False)  # Time vector
t = np.linspace(0, 1, fs, endpoint=False)
frequencies = [10, 50, 200]  # Frequencies in the signal
signal_data = np.sin(2 * np.pi * frequencies[0] * t) + \
              np.sin(2 * np.pi * frequencies[1] * t) + \
              np.sin(2 * np.pi * frequencies[2] * t)

# Perform FFT
fft_result = np.fft.fft(signal_data)
fft_freqs = np.fft.fftfreq(len(t), 1/fs)

# Perform STFT
f, t, stft_result = signal.stft(signal_data, fs=fs, nperseg=100, noverlap=50)

# Perform Wavelet Transform (using Daubechies wavelet)
wavelet_result, _ = pywt.cwt(signal_data, scales=np.arange(1, 31), wavelet='cmor')

# Plotting
plt.figure(figsize=(12, 8))

# Original Signal
plt.subplot(4, 1, 1)
plt.plot(t1, signal_data)
plt.title('(a) Original Signal')

# FFT
plt.subplot(4, 1, 2)
plt.plot(fft_freqs, np.abs(fft_result))
plt.title('(b) FFT')

# STFT
plt.subplot(4, 1, 3)
#plt.plot(f, np.abs(stft_result))
plt.pcolormesh(t, f, np.abs(stft_result), shading='auto')
plt.title('(c) STFT')
plt.ylabel('Frequency (Hz)')

# Wavelet Transform
plt.subplot(4, 1, 4)
plt.imshow(np.abs(wavelet_result), extent=[0, len(t), 1, 31], cmap='viridis', aspect='auto')
#plt.plot(_ , np.abs(wavelet_result))
plt.title('(d) Wavelet Transform')
plt.xlabel('Time')
plt.ylabel('Scale')

plt.tight_layout()
plt.show()
