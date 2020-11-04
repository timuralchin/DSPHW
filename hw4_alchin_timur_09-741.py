import scipy.io.wavfile as wav
import numpy as np
import matplotlib.pyplot as plt


class Watermarking:
    '''
    Watermark class for watermark generating based on
    Irreducible polynomial
    '''

    def __init__(self, s0, xn, D):
        '''

        :param s0: base vector
        :param xn: Irreducible polynomial
        :param D: LPM vector
        '''
        self.s0 = s0
        self.xn = xn
        self.M = 2 ** (len(self.xn) - 1) - 1  # period
        self.D = D
        self.watermark = []  # watermark initialization
        self.A = self.conpanion_matrix()

    def conpanion_matrix(self):
        '''

        :return: conpanion matrix for irreducible polynomial
        '''
        A = []
        for i in range(len(self.xn) - 2):
            row = np.zeros(len(self.xn) - 1)
            row[i + 1] = 1
            A.append(row.astype(int))
        A.append(self.xn[:len(self.xn) - 1])
        return np.array(A)

    def create_watermark(self):
        '''

        :return: watermark
        '''
        s0 = self.s0
        for i in range(self.M):
            s = (self.A ** (i + 1) @ s0)
            s0 = s % 2
            w = np.dot(self.D, s) % 2
            self.watermark.append(w)

        for i in range(len(self.watermark)):
            if self.watermark[i] == 0:
                self.watermark[i] = -1

        return self.watermark


def main():
    Xn = [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]
    S0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    D = Xn.copy()[:len(Xn) - 1]
    watermark = Watermarking(S0, Xn, D)
    watermark = watermark.create_watermark()
    fs, data = wav.read('watermark.wav')
    h = np.correlate(data, watermark)
    p = np.argmax(h)
    print("Watermark position: " + str(p) + " samples")
    print(str(p / fs) + " seconds")
    timelime = np.linspace(0, len(h) / fs, len(h))
    plt.plot(timelime, h)
    plt.show()


if __name__ == "__main__":
    main()
