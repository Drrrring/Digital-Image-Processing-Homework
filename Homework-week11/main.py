import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('Resource/cover1.jpg', cv2.IMREAD_GRAYSCALE)

# resize the image because it's too big
# width = img.shape[1] // 2
# height = img.shape[0] // 2
# dim = (width, height)
# img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

# apply motion blur
kernelSize = 19
kernel = np.zeros([kernelSize, kernelSize])
for i in range(kernelSize):
    kernel[i, i] = 1
kernel /= kernelSize
imgMotionBlur = cv2.filter2D(img, -1, kernel)

# add noise
row, col = imgMotionBlur.shape
gaussNoise = np.zeros((row, col), dtype=np.uint8)
# with a mean of 128 and a sigma of 20
cv2.randn(gaussNoise, 128, 0.1)
gaussNoise = (gaussNoise * 0.5).astype(np.uint8)
imgCorrupted = cv2.add(imgMotionBlur, gaussNoise)

# apply inverse filter
# inverseFilterRes = scipy.signal.deconvolve(imgCorrupted, kernel)
P = img.shape[0] + kernelSize - 1
if P % 2 == 1:
    P = P + 1
Q = img.shape[1] + kernelSize - 1
if Q % 2 == 1:
    Q = Q + 1

padded = np.pad(img, ((0, P - img.shape[0]), (0, Q - img.shape[1])), 'constant', constant_values=(0, 0))
# dft = np.fft.fft2(padded)
# dft_shift = np.fft.fftshift(dft)
# plt.imshow(dft_shift)
# magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1)

padded_kernel = np.pad(kernel, ((1, P - kernelSize - 1), (1, Q - kernelSize - 1)), 'constant', constant_values=(0, 0))

fft_kernel = np.fft.fft2(padded_kernel)
fft_kernel_shift = np.fft.fftshift(fft_kernel)
fft_img = np.fft.fft2(padded)
fft_img_shift = np.fft.fftshift(fft_img)

# test
# a = 20*np.log(np.abs(fft_img_shift)+1)
# plt.imshow(a, cmap='gray')
# plt.show()
# b = 20*np.log(np.abs(fft_kernel_shift)+1)
# plt.imshow(b, cmap='gray')
# plt.show()

divRes_shift = fft_img_shift / fft_kernel_shift
divRes_ishift = np.fft.ifftshift(divRes_shift)
divRes = np.fft.ifft2(divRes_ishift)
divRes = np.abs(divRes)
divRes = np.uint8(divRes)

# wiener
K = 0.35
fft_img_power = np.absolute(fft_kernel) ** 2
wiener_fre = fft_img_power * fft_img / fft_kernel / (fft_img_power + K)
wienerRes = np.fft.ifft2(wiener_fre)
wienerRes = np.abs(wienerRes)
wienerRes = np.uint8(wienerRes)

plt.subplot(1, 3, 1), plt.imshow(imgCorrupted, cmap='gray', vmin=0, vmax=255)
plt.title('with motion blur and gaussian noise'), plt.xticks([]), plt.yticks([])

plt.subplot(1, 3, 2), plt.imshow(divRes, cmap='gray', vmin=0, vmax=255)
plt.title('inverse filter'), plt.xticks([]), plt.yticks([])

plt.subplot(1, 3, 3), plt.imshow(wienerRes, cmap='gray', vmin=0, vmax=255)
plt.title('wiener'), plt.xticks([]), plt.yticks([])

plt.show()
# cv2.imshow('original', img)
# cv2.imshow('motion blur', imgMotionBlur)
# cv2.imshow('corrupt', imgCorrupted)
# cv2.waitKey(0)
