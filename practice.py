import cv2
from numpy import fft
import numpy as np
import matplotlib.pyplot as plt

# h = 1/4*np.array([1, 2, 1])
# h = np.pad(h, (126, 127), 'constant')

# x = [_ - 127 for _ in range(256)]

# Hk = fft.fft(h)
# Hk = fft.fftshift(Hk)

# h = 1/4*np.array([-1, 2, -1])
# h = np.pad(h, (126, 127), 'constant')

# x = [_ -127 for _ in range(256)]

# Hk = fft.fft(h)
# Hk = fft.fftshift(Hk)

# plt.figure(figsize=(8, 2))
# plt.plot(x, np.abs(Hk))
# plt.show()

in_image_dgu = cv2.imread('dgu_gray.png', 0)

f_dgu = fft.fft2(in_image_dgu)
fshift_dgu = fft.fftshift(f_dgu)
magnitude_dgu = 20*np.log(np.abs(fshift_dgu))
phase_dgu = np.angle(fshift_dgu)

plt.subplot(234), plt.imshow(in_image_dgu, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(232), plt.imshow(magnitude_dgu, cmap='gray')
plt.title('Magnitude'), plt.xticks([]), plt.yticks([])
plt.subplot(233), plt.imshow(phase_dgu, cmap='gray')
plt.title('Phase'), plt.xticks([]), plt.yticks([])

plt.show()
