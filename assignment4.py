import cv2
from numpy import fft
import numpy as np
import matplotlib.pyplot as plt

# h = 1/4*np.array([1, 2, 1])
# h = np.pad(h, (126, 127), 'constant')

# x = [_ - 127 for _ in range(256)]

# Hk = fft.fft(h)
# Hk = fft.fftshift(Hk)

h = 1/4*np.array([-1, 2, -1])
h = np.pad(h, (126, 127), 'constant')

x = [_ -127 for _ in range(256)]

Hk = fft.fft(h)
Hk = fft.fftshift(Hk)

plt.figure(figsize=(8, 2))
plt.plot(x, np.abs(Hk))
plt.show()
