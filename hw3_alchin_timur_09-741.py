import scipy.io.wavfile as wav
import numpy as np
import scipy as sc
import scipy.signal as sig
import matplotlib.pyplot as plt


class myarray(np.ndarray):
    """
    class for finding element in np.array by index
    """
    def __new__(cls, *args, **kwargs):
        return np.array(*args, **kwargs).view(myarray)
    def index(self, value):
        return np.where(self == value)

class signal:

    def __init__(self, filepath='voice.wav'):
        """

        :param filepath: input file path
        :var fs: sampling rate
        :var data: signal data
        """
        self.filepath=filepath
        self.fs, self.data = wav.read(self.filepath)
        self.len = len(self.data)
        self.time = self.len / self.fs

    def show_spectrum(self, start_time=0, end_time=1):
        """

        :param start_time: spectrum part start time
        :param end_time: spectrum part end time
        :return none
        """
        signal = self.data[int(start_time * self.fs): int(end_time * self.fs)]
        dft_signal, samples, freq = self.calc_spectrum(signal)
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
        max1 = max(abs(dft_signal))
        z = myarray(abs(dft_signal))
        t = z.index(max1)
        print("The largest spectrum value frequency: " + str(freq[t]) + " HZ")

    def calc_spectrum(self, signal):
        """

        :param signal: input signal part
        :return: dft array, signal len, frequency array
        """
        samples = len(signal)
        samples_array = np.arange(samples)
        local_time=samples / self.fs
        freq = (samples_array / local_time)[range(int(samples / 2))]

        hann_window = sig.windows.hann(samples) #hann window function for correct fft computation
        windowed_signal = signal * hann_window #signal part and hann multplication
        dft_signal = (sc.fft.fft(windowed_signal)/samples)[range(int(samples/ 2))] #dft signal computation

        return dft_signal,samples, freq


    def show_spectrogram(self,start_time=0, end_time=1):
        """

        :param start_time: spectrum part start time
        :param end_time: spectrum part end time
        :return: none
        """
        spectrogram = [] #fft arrays list
        signal = self.data[int(start_time * self.fs): int(end_time * self.fs)] #siganl part
        signal_len = len(signal)
        step = 0.005 * self.fs #window step
        window = 0.01 * self.fs #window range
        left_side = 0
        right_side = window
        time_spectrogram=[]
        t=0

        while right_side < signal_len:
            dft_signal, samples, freq = self.calc_spectrum(signal[left_side:int(right_side)])
            spectrogram.append(dft_signal)
            left_side += int(step)
            right_side += int(step)
            t+=int(step)
            time_spectrogram.append(t)

        img = np.zeros((len(spectrogram[0]),len(time_spectrogram), 3), np.uint8) #spectrogram image matrix

        max1=0
        for spec in spectrogram:
            if max(spec)> max1:
                max1 = max(spec)

        k=255/max1 #coef to normalize

        for i in range(len(time_spectrogram) - 1):
            for j in range(len(spectrogram[0]) - 1):
                point = abs(spectrogram[i][j])
                img[len(spectrogram[0]) - 1 - j][i] = (point * k, point * k, point * k)

        plt.imshow(img)
        plt.show()

def main():
    VOICE_START,VOICE_END = 1.7,4
    NOISE_START, NOISE_END = 4,6
    voice = signal()
    voice.show_spectrum(VOICE_START, VOICE_END)
    noise = signal()
    noise.show_spectrum(NOISE_START,NOISE_END)
    voice.show_spectrogram(VOICE_START,VOICE_END)

if __name__=="__main__":
    main()
