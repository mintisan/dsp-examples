# http://ataspinar.com/2018/04/04/machine-learning-with-signal-processing-techniques/
# The auto-correlation function calculates the correlation of a signal with a time-delayed version of itself.
# The idea behind it is that if a signal contain a pattern which repeats itself after a time-period of \tau seconds,
# there will be a high correlation between the signal and a \tau sec delayed version of the signal.

import numpy as np
import matplotlib.pyplot as plt

t_n = 10
N = 1000
T = t_n / N
f_s = 1 / T

x_value = np.linspace(0, t_n, N)
amplitudes = [4, 6, 8, 10, 14]
frequencies = [6.5, 5, 3, 1.5, 1]
y_values = [amplitudes[ii] * np.sin(2 * np.pi * frequencies[ii] * x_value) for ii in range(0, len(amplitudes))]
composite_y_value = np.sum(y_values, axis=0)

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[len(result)//2:]

def get_autocorr_values(y_values, T, N, f_s):
    autocorr_values = autocorr(y_values)
    x_values = np.array([T * jj for jj in range(0, N)])
    return x_values, autocorr_values

t_n = 10
N = 1000
T = t_n / N
f_s = 1/T

t_values, autocorr_values = get_autocorr_values(composite_y_value, T, N, f_s)

plt.plot(t_values, autocorr_values, linestyle='-', color='blue')
plt.xlabel('time delay [s]')
plt.ylabel('Autocorrelation amplitude')
plt.show()