import scipy.io.wavfile as wav
import numpy as np
from bitarray import bitarray


class Watermark:
    '''
    Watermark class for audio file watermarking
    '''

    def __init__(self, watermark, A=0.3, time_parse_len=0.01):
        """

        :param watermark: <str> watermark
        :param input_file_path: <str> input audiofile path
        :param A: <float> "A" parameter,should be betwenn 0.1 and 0.5
        :param time_parse_len: <float> Time windows len
        """
        bit_watermark = bitarray()
        bit_watermark.frombytes(watermark.encode('utf-8'))
        self.watermark = bit_watermark.to01()
        self.A = A
        self.time_parse_len = time_parse_len



    def __u(self, bit, t):
        """

        :param bit: bit for definition of u function sign
        :param t: input value
        :return: u function value
        """
        res = 1 + self.A * self.__spl(t) if bit else 1 - self.A * self.__spl(t)
        return res

    def watermarking(self, input_file_path, watermark_start_pos):
        """

        :param watermark_start_pos: start position for watermark implantation
        :return: watermark, watermark start position, watermark end position
        """
        self.input_file_path = input_file_path
        self.fs, self.data = wav.read(self.input_file_path)
        #exception handling
        if watermark_start_pos * self.fs < 0 or watermark_start_pos * self.fs >= len(
                self.data) or watermark_start_pos * self.fs + len(self.watermark) >= len(self.data):
            return print("Watermark position error! Choose another one!")

        start_position = int(watermark_start_pos * self.fs)
        end_position = start_position + int(len(self.watermark) * self.time_parse_len * self.fs)
        data_for_watermark = self.data[start_position: end_position]
        parse_len = int(self.time_parse_len * self.fs)

        parsed_data_for_watermark = self.__parse_data(data_for_watermark, parse_len)
        watermarked_parsed_data = []

        for i in range(len(self.watermark)):
            for j in range(parse_len):
                watermarked_parsed_data.append(
                    parsed_data_for_watermark[i][j] * self.__u(int(self.watermark[i]), j / parse_len))

        watermarked_data = np.array(self.data)
        watermarked_parsed_data = np.array(watermarked_parsed_data)
        watermarked_parsed_data = watermarked_parsed_data.astype(int)

        watermarked_data[start_position:end_position] = watermarked_parsed_data

        return watermarked_data, start_position, end_position

    def get_watermark(self, watermarked_data, start_position, end_position):
        """

        :param watermarked_data: watermarked signal
        :param start_position: watermark start position
        :param end_position: watermark end position
        :return: original watermark
        """
        watermark = ''
        parsed_watermarked_data = self.__parse_data(watermarked_data[start_position:end_position],
                                                  int(self.time_parse_len * self.fs))
        original_parsed_data = self.__parse_data(self.data[start_position:end_position],
                                               int(self.time_parse_len * self.fs))
        for i in range(len(parsed_watermarked_data)):
            if self.__signal_strength(parsed_watermarked_data[i]) > self.__signal_strength(original_parsed_data[i]):
                watermark += '1'
            else:
                watermark += '0'

        return bitarray(watermark).tobytes().decode()

    def __parse_data(self, data, parse_len):
        """

        :param data: data for parsing
        :param parse_len: len of parsing
        :return: parsed data
        """
        parsed_data = []
        data = data
        for i in range(len(self.watermark)):
            parsed_data.append(data[i * parse_len:(i + 1) * parse_len])
        return parsed_data

    def add_noise(self,data):
        noise = np.random.normal(0,1,len(data))
        return data + noise


    @staticmethod
    def __spl(t):
        if 0 <= t <= 1 / 3:
            return 9 / 2 * (t ** 2)
        if 1 / 3 < t <= 2 / 3:
            return (-9 * (t ** 2)) + (9 * t) - 3 / 2
        if 2 / 3 < t <= 1:
            return 9 / 2 * ((1 - t) ** 2)

    @staticmethod
    def __signal_strength(data):
        strength = 0
        for i in data:
            strength += i ** 2

        return strength


def main():
    print("Type your watrmark:")
    watermark = input()
    print("Type implantation position (sec):")
    position = input()
    watermark = Watermark(watermark, A=0.5)
    watermarked_signal, start, end = watermark.watermarking('voice.wav', int(position))
    print(f'Your watermark:{watermark.get_watermark(watermarked_signal, start, end)}')
    noised_signal = watermark.add_noise(watermarked_signal)
    print(f'Your watermark after noise addition:{watermark.get_watermark(noised_signal, start, end)}')


if __name__ == "__main__":
    main()
