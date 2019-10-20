# http://ataspinar.com/2018/04/04/machine-learning-with-signal-processing-techniques/
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

t_n = 10
N = 1000
T = t_n / N
f_s = 1 / T

x_value = np.linspace(0, t_n, N)
amplitudes = [4, 6, 8, 10, 14]
frequencies = [6.5, 5, 3, 1.5, 1]
y_values = [amplitudes[ii] * np.sin(2 * np.pi * frequencies[ii] * x_value) for ii in range(0, len(amplitudes))]
composite_y_value = np.sum(y_values, axis=0)

def get_psd_values(y_values, T, N, f_s):
    f_values, psd_values = welch(y_values, fs=f_s)
    return f_values, psd_values


t_n = 10
N = 1000
T = t_n / N
f_s = 1/T

f_values, psd_values = get_psd_values(composite_y_value, T, N, f_s)

plt.plot(f_values, psd_values, linestyle='-', color='blue')
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [V**2 / Hz]')
plt.show()