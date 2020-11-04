from bitarray import bitarray

import scipy.io.wavfile as wav
import numpy as np
from pydub import AudioSegment

inp = 'watermark.wav'
out = 'watermark.mp3'

s = AudioSegment.from_wav(inp)
s.export(out, format = "mp3")