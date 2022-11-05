import cv2
import numpy as np

img = cv2.imread('dgu_gray.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 이미지를 흑백으로 변환
height, width = gray.shape

mask1 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
mask2 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
mask3 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

laplacian1 = cv2.filter2D(gray, -1, mask1)
laplacian2 = cv2.filter2D(gray, -1, mask2)
laplacian3 = cv2.filter2D(gray, -1, mask3)
laplacian4 = cv2.Laplacian(gray, -1)

# LoG
gaussian = cv2.GaussianBlur(gray, (5, 5), 0)
LoG = cv2.filter2D(gaussian, -1, mask3)


# Difference of Gaussian
gaussian1 = cv2.GaussianBlur(gray, (5, 5), 1.6)
gaussian2 = cv2.GaussianBlur(gray, (5, 5), 1)
DoG = np.zeros_like(gray)
for i in range(height):
    for j in range(width):
        DoG[i][j] = float(gaussian1[i][j]) - float(gaussian2[i][j])


cv2.imshow('original', gray)
cv2.imshow('laplacian1', laplacian1.astype(np.float))
cv2.imshow('laplacian2', laplacian2.astype(np.float))
cv2.imshow('laplacian3', laplacian3.astype(np.float))
cv2.imshow('laplacian4', laplacian3.astype(np.float))
#cv2.imshow('gaussian', gaussian)
cv2.imshow('LoG', LoG.astype(np.float))
cv2.imshow('gaussian1', gaussian1)
cv2.imshow('gaussian2', gaussian2)
cv2.imshow('DoG', DoG)
cv2.waitKey(0)

