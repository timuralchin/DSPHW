from bitarray import bitarray

import scipy.io.wavfile as wav
import numpy as np
from pydub import AudioSegment
import ffmpeg
import os

msg = "somebody once told me the world is gonna roll me, i ain't the sharpest tool"
bar = bitarray()
bar.frombytes(msg.encode('utf-8'))
#print(bar)
sbit = bar.to01()  # text string bitarray
msg_decodet = bar.tobytes().decode()  # return to string
fs, data = wav.read('watermark.wav')
A = 0.5



def u(bit, t):
    res = 1 + A * spl(t) if bit else 1 - A * spl(t)
    return res


def spl(t):
    if 0 <= t <= 1 / 3:
        return 9 / 2 * (t ** 2)
    if 1 / 3 < t <= 2 / 3:
        return (-9 * (t ** 2)) + (9 * t) - 3 / 2
    if 2 / 3 < t <= 1:
        return 9 / 2 * ((1 - t) ** 2)


#print(fs, np.around(len(data) / fs))
data = data[:fs * 6]
part = int(fs * 0.01)
#print(np.around(len(data) / fs) / 0.01, len(sbit))
data_parts = []
for i in range(len(sbit)):
    data_parts.append(data[int(i * part):(i + 1) * part])

new_data = []
new_data_parts = []
for i in range(len(sbit)):
    raw = []
    for j in range(part):
        new_data.append(data_parts[i][j] * u(int(sbit[i]), j / part))
        raw.append(data_parts[i][j] * u(int(sbit[i]), j / part))
    new_data_parts.append(raw)

def power(array):
    s=0
    for i in array:
        s+=i**2
    return s
new_data = np.array(new_data)
new_data =new_data.astype(int)

wav.write('input_files/voice_water.wav',fs, new_data)
fs, new_data=wav.read('input_files/voice_water.wav')
new_datalist=[]


print(len(sbit), len(data_parts))
for i in range(len(sbit)):
    new_datalist.append(new_data[int(i * part):(i + 1) * part])
binmsg = ''
for i in range(len(new_data_parts)):
    if power(new_datalist[i])>power(data_parts[i]):
        binmsg+='1'
    else:
        binmsg += '0'
new_data=np.array(new_data)
binmsg = bitarray(binmsg)

print(binmsg.tobytes().decode())
