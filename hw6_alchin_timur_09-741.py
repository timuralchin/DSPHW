import numpy as np
import scipy.signal as sgn
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from numpy.fft import rfft, rfftfreq
import scipy as sc

def calc_spectrum(signal, fs):
    samples = len(signal)
    samples_array = np.arange(samples)
    local_time = samples / fs
    freq = (samples_array / local_time)[range(int(samples/2))]

    hann_window = sgn.windows.hann(samples)  # hann window function for correct fft computation
    windowed_signal = signal * hann_window  # signal part and hann multplication
    dft_signal = (sc.fft.fft(windowed_signal) / samples)[range(int(samples/2))]  # dft signal computation

    return dft_signal, samples, freq

def show(signal,fs):
    dft_signal, samples, freq = calc_spectrum(signal,fs)
    plot_timeline = np.arange(0, samples, 1)
    plt.subplot(2, 1, 1)
    plt.plot(plot_timeline, signal)
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.subplot(2, 1, 2)
    plt.plot(freq, abs(dft_signal), 'g')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.show()

class Filter:
    def __init__(self, order = 20, freqs = [0.3, 0.7] ):
        self.freqs = freqs
        self.order = order

    def filter(self, signal):
        b0 = sgn.firwin(self.order, self.freqs, pass_zero="bandpass")
        b1, a1 = sgn.butter(self.order, self.freqs, btype="bandstop")
        w0, h0 = sgn.freqz(b0)
        w1, h1 = sgn.freqz(b1,a1)
        h = h0 * h1
        filtred_signal_fir = sgn.filtfilt(b0,1,signal)
        filtred_signal_iir = sgn.filtfilt(b1,a1,filtred_signal_fir)

        return filtred_signal_iir, w0,h0,w1,h1,h
fs = 200
freqs = [15,30,50,75,90]
time=1
t = np.linspace(0, time, int(time*fs))
signal = 0
for freq in freqs:
    signal+=np.sin(2*np.pi*freq*t)
show(signal,fs)
powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(signal, Fs=fs)
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.show()
plt.plot(signal)

filt = Filter()
signal, w0,h0,w1,h1,h = filt.filter(signal)
plt.plot(signal)
plt.show()
plt.plot(100 * w1/ np.pi, abs(h))
plt.plot(100 * w1 / np.pi, abs(h1))
plt.plot(100 * w1 / np.pi, abs(h0))

plt.show()
show(signal,fs)
powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(signal, Fs=fs)
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.show()
