import jadeR
import numpy as np


class ICA(object):
    def __init__(self):
        self.components = None

    def analyse_rgb_channels(self, rgb_avgs, no_of_examples):
        np_rgb = np.ndarray(shape=(3, no_of_examples), buffer=np.array(rgb_avgs))
        np_rgb = self.normalise_values(np_rgb)
        return jadeR.main(np_rgb)

    def normalise_values(self, values):
        for array in values:
            self.normalise_array(array)
        return values

    @staticmethod
    def normalise_array(array):
        mean = np.mean(array)
        std = np.std(array)

        for i in range(len(array)):
            array[i] = ((array[i] - mean) / std)
        return array

    # noinspection PyTypeChecker,PyTypeChecker
    @staticmethod
    def select_pertinent_signal(signals):
        signal1 = np.squeeze(np.asarray(signals[:, 0])).tolist()
        signal2 = np.squeeze(np.asarray(signals[:, 1])).tolist()
        signal3 = np.squeeze(np.asarray(signals[:, 2])).tolist()

        comp1 = np.hamming(len(signal1)) * signal1
        comp2 = np.hamming(len(signal2)) * signal2
        comp3 = np.hamming(len(signal3)) * signal3

        comp1 = np.abs(np.square(np.fft.irfft(comp1))).astype(float).tolist()
        comp2 = np.abs(np.square(np.fft.irfft(comp2))).astype(float).tolist()
        comp3 = np.abs(np.square(np.fft.irfft(comp3))).astype(float).tolist()

        power_ratio1 = np.sum(comp1)/np.amax(comp1)
        power_ratio2 = np.sum(comp2)/np.amax(comp2)
        power_ratio3 = np.sum(comp3)/np.amax(comp3)

        if power_ratio1 > power_ratio2 and power_ratio1 > power_ratio2:
            return {'signal': signal1, 'component': comp1}
        elif power_ratio2 > power_ratio1 and power_ratio2 > power_ratio3:
            return {'signal': signal2, 'component': comp2}
        else:
            return {'signal': signal3, 'component': comp3}
