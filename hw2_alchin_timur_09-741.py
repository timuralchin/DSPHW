import numpy as np
import matplotlib.pyplot as plt

class AnalogSinSignal():
    def __init__(self, frequency=10, sampling_rate=1000, time=1):
        self.frequency=frequency
        self.sampling_rate=sampling_rate
        self.time=time
        self.T_sample = 1 / self.sampling_rate
        self.samples_num=np.linspace(0, self.time, self.sampling_rate * self.time)
        self.signal=np.sin(2 * np.pi * self.frequency * self.samples_num)

    def show(self ):
        plt.figure()
        plt.plot(self.signal)
        plt.xlabel("Семплы")
        plt.ylabel("Амплитуда")
        plt.stem(self.signal)
        plt.show()

class  SampledSinSignal(AnalogSinSignal):

    def sinc(self, samples):
        res=np.sin((samples * np.pi)/ self.T_sample) / (samples * np.pi)
        return res

    def resamplig(self,samples):
        sum=0
        length=len(self.signal)
        for i in range (length):
            sum+=self.signal[i]*self.sinc(samples-self.samples_num[i])
        sum=sum * self.T_sample
        return sum

    def recover(self, samples):
        recovered_signal=[]
        for i in samples:
            recovered_signal.append(self.resamplig(i))
        return recovered_signal

def main():
    original_signal=AnalogSinSignal()
    bad_signal=SampledSinSignal(sampling_rate=10)
    good_siganl=SampledSinSignal(sampling_rate=30)
    bad_recovered_signal=bad_signal.recover(original_signal.samples_num)
    good_recovered_signal=good_siganl.recover(original_signal.samples_num)
    plt.figure()
    plt.plot(original_signal.signal, color="blue")
    plt.plot(bad_recovered_signal, color="red")
    plt.plot(good_recovered_signal, color="green")
    plt.xlabel("Семплы")
    plt.ylabel("Амплитуда")
    plt.show()

if __name__=="__main__":
    main()




