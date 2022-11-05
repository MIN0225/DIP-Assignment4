import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy import fft

in_image_dgu = cv2.imread('dgu_gray.png', 0)
height, width = in_image_dgu.shape

k=50
h = np.ones((k, k))
h_lp = np.pad(h, ((height//2-k//2, width//2-k//2), (height//2-k//2, width//2-k//2)), 'constant')
h_hp = np.ones((height, width)) - h_lp

f_dgu = fft.fft2(in_image_dgu)
fshift_dgu = fft.fftshift(f_dgu)
magnitude_dgu = 20 * np.log(np.abs(fshift_dgu))
phase_dgu = np.angle(fshift_dgu)

fshift_dgu_lp = np.multiply(np.abs(fshift_dgu), h_lp)
fshift_dgu_hp = np.multiply(np.abs(fshift_dgu), h_hp)

recon_dgu_lp = np.multiply(fshift_dgu_lp, np.exp(1j * phase_dgu))
img_recon_dgu_lp = np.minimum(np.abs(np.real(fft.ifft2(fft.fftshift(recon_dgu_lp)))), 255)
recon_dgu_hp = np.multiply(fshift_dgu_hp, np.exp(1j * phase_dgu))
img_recon_dgu_hp = np.minimum(np.abs(np.real(fft.ifft2(fft.fftshift(recon_dgu_hp)))), 255)


# plt.subplot(231), plt.imshow(in_image_dgu, cmap='gray') # plt.subplot(ijk) i행 j열 구조. k번째에 배치
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])

plt.subplot(234), plt.imshow(img_recon_dgu_lp, cmap='gray')
plt.title('recon_dgu_lp'), plt.xticks([]), plt.yticks([])
plt.subplot(235), plt.imshow(img_recon_dgu_hp, cmap='gray')
plt.title('recon_dgu_lp'), plt.xticks([]), plt.yticks([])

plt.subplot(232), plt.imshow(magnitude_dgu, cmap='gray')
plt.title('Magnitude'), plt.xticks([]), plt.yticks([])
plt.subplot(233), plt.imshow(phase_dgu, cmap='gray')
plt.title('Phase'), plt.xticks([]), plt.yticks([])

plt.show()
