import cv2
import numpy as np
from matplotlib import pyplot as plt

img0 = cv2.imread('Resource/building.jpg', )

# converting to gray scale
gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

# convolution sobel kernel
sobelKernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
# paddedSobelKernel = np.array([[0, 0, 0, 0], [0, -1, 0, 1], [0, -2, 0, 2], [0, -1, 0, 1]])
# paddedSobelKernel = np.zeros([340, 340])
# paddedSobelKernel[1, 1] = -1
# paddedSobelKernel[1, 3] = 1
# paddedSobelKernel[2, 1] = -2
# paddedSobelKernel[2, 3] = 2
# paddedSobelKernel[3, 1] = -1
# paddedSobelKernel[3, 3] = 1
#
# addColumns = np.zeros([338, 2])
# addRows = np.zeros(([2, 340]))
#
# paddedGray = np.append(gray, addColumns, axis=1)
# paddedGray = np.append(paddedGray, addRows, axis=0)

# 搬移
# for x in range(340):
#     for y in range(340):
#         paddedSobelKernel[x, y] *= pow(-1, x + y)
#
# paddedKernel = cv2.dft(paddedSobelKernel, flags=cv2.DFT_COMPLEX_OUTPUT)
# for u in range(340):
#     for v in range(340):
#         # 实部设置为0
#         paddedKernel[u, v, 0] = 0
#         # dftKernel[u, v, 1] *= pow(-1, u + v)
#
# dftGray = cv2.dft(paddedGray, flags=cv2.DFT_COMPLEX_OUTPUT)
# freqResult = cv2.mulSpectrums(dftGray, paddedKernel, flags=cv2.DFT_COMPLEX_OUTPUT)
# freqResult = cv2.dft(freqResult, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT + cv2.DFT_INVERSE)

spatialResult = np.zeros([338, 338])
grayPadded = np.pad(gray, pad_width=1, mode='constant')
for x in range(338):
    for y in range(338):
        add = sobelKernel[0][0] * grayPadded[x][y] + sobelKernel[0][1] * grayPadded[x][y + 1] + sobelKernel[0][2] * \
              grayPadded[x][y + 2] + sobelKernel[1][0] * grayPadded[x + 1][y] + sobelKernel[1][1] * grayPadded[x + 1][
                  y + 1] + sobelKernel[1][2] * grayPadded[x + 1][y + 2] + sobelKernel[2][0] * grayPadded[x + 2][y + 0] + \
              sobelKernel[2][1] * grayPadded[x + 2][y + 1] + sobelKernel[2][2] * grayPadded[x + 2][y + 2]
        spatialResult[x][y] = add


# Padding Images with zeros
padding=[340, 340]
paddedImage = np.zeros(padding, dtype="float32")
paddedImage[:gray.shape[0], :gray.shape[1]] = gray
paddedKernel = np.zeros(padding, dtype="float32")
paddedKernel[:sobelKernel.shape[0], :sobelKernel.shape[1]] = sobelKernel
# DFT
dftImage = cv2.dft(paddedImage)
dftKernel = cv2.dft(paddedKernel)
# Multiplication
mulRes = cv2.mulSpectrums(dftImage, dftKernel, flags=cv2.DFT_COMPLEX_OUTPUT)
# Get Result
frequencyResult = cv2.idft(mulRes, flags=cv2.DFT_SCALE)

# spatial = cv2.filter2D(src=gray, ddepth=-1, kernel=sobelKernel)

# sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # x

# sz = (gray.shape[0] - sobelKernel.shape[0], gray.shape[1] - sobelKernel.shape[1])  # total amount of padding
# paddedKernel = np.pad(sobelKernel, (((sz[0] + 1) // 2, sz[0] // 2), ((sz[1] + 1) // 2, sz[1] // 2)), 'constant')
#
# filtered = np.real(fftpack.ifft2(fftpack.fft2(gray) * fftpack.fft2(paddedKernel))) + np.imag(
#     fftpack.ifft2(fftpack.fft2(gray) * fftpack.fft2(paddedKernel)))
# filtered = np.maximum(0, np.minimum(filtered, 255))

plt.subplot(1, 3, 1), plt.imshow(gray, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(1, 3, 2), plt.imshow(frequencyResult, cmap='gray', vmin=-256, vmax=255)
plt.title('dft'), plt.xticks([]), plt.yticks([])

plt.subplot(1, 3, 3), plt.imshow(spatialResult, cmap='gray', vmin=-256, vmax=255)
plt.title('Spatial'), plt.xticks([]), plt.yticks([])

plt.show()


# def logMagnitude(filter):
#     freq = np.fft.fft2(filter)
#     transfreq = np.fft.fftshift(freq)
#     return np.log(1 + abs(transfreq))
