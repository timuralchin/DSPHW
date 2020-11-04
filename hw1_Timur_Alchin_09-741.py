import numpy as np
import matplotlib.pyplot as plt

#создание гармонического сигнала
def create_sin_signal( fs,freq,time=1):
    timeline=np.linspace(0,time,time*fs)
    signal=np.sin(2*np.pi*freq*timeline)
    return signal

#вычисление шага квантования
def get_quant_step(bit, M1=1,M2=-1):
    h=(M1-M2)/2**(bit)
    return h

#квантование сигнала
def quant_signal(signal, q):
    qs=q*np.round(signal/q)
    return qs

#вычисление шага квантования
def calc_noise(signal, qs):
    e=signal-qs
    return e

#создание случайного сигнала
def create_sig_random(fs,time=1):
    sig=np.random.rand(time*fs)
    return sig

#вывод сигнала
def show_signal(signals):
    plt.figure()
    for signal in signals:
        plt.plot(signal)
    plt.show()

#вычисление дисперсии
def disp(signal,e):
    disps=[np.var(signal),np.var(e)]
    return disps

def main(bit):
    sin_signal=create_sin_signal(1000,10)
    rand_signal=create_sig_random(1000)

    quant_sin=quant_signal(sin_signal, get_quant_step(bit))
    quant_rand=quant_signal(rand_signal, get_quant_step(bit))

    t_snr1=6*bit-7.2

    disps_sin = disp(sin_signal,calc_noise(sin_signal,quant_sin))
    disps_rand = disp(rand_signal, calc_noise(rand_signal, quant_rand))

    snr_sin=10*np.log10(disps_sin[0]/disps_sin[1])
    snr_rand=10*np.log10(disps_rand[0]/disps_rand[1])

    print("Теоретическое snr: " + str(t_snr1))
    print("Полученное snr синус: " + str(snr_sin))
    print("Полученное snr рандом: " + str(snr_rand))

if __name__=="__main__":
    bit=16
    main(bit)


