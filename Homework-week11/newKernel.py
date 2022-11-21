import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('Resource/cover1.jpg', cv2.IMREAD_GRAYSCALE)

# apply motion blur
kernel = np.zeros(img.shape)
center_position = (img.shape[0] - 1) // 2
slope_tan = 1
distance = 16
for i in range(distance):
    offset = round(i * slope_tan)
    kernel[int(center_position + offset), int(center_position + offset)] = 1
kernel /= kernel.sum()
img_fft = np.fft.fft2(img)
kernel_fft = np.fft.fft2(kernel)
blurred = np.fft.ifft2(img_fft * kernel_fft)
blurred = np.abs(np.fft.fftshift(blurred))

# add gaussian noise
def add_gaussian_noise(img, sigma):
    if sigma < 0:
        return img
    gauss = np.random.normal(0, sigma, np.shape(img))
    noisy_img = img + gauss
    noisy_img[noisy_img < 0] = 0
    noisy_img[noisy_img > 255] = 255
    return noisy_img


# inverse filter
def inverse_filter(img, kernel):
    img_fft = np.fft.fft2(img)
    kernel_fft = np.fft.fft2(kernel) + 0.001
    result = np.fft.ifft2(img_fft / kernel_fft)
    result = np.abs(np.fft.fftshift(result))
    return result

# wiener filter
def wiener_filter(img, kernel, K=0.01):
    img_fft = np.fft.fft2(img)
    kernel_fft = np.fft.fft2(kernel)
    kernel_fft_1 = np.conj(kernel_fft) / (np.abs(kernel_fft) ** 2 + K)
    result = np.fft.ifft2(img_fft * kernel_fft_1)
    result = np.abs(np.fft.fftshift(result))
    return result


sigmas = [50,10,1]
ks = [0.1,0.01,0.0008]
for i in range(len(sigmas)):
    # corrupted image
    img_corrupted = add_gaussian_noise(blurred, sigmas[i])
    plt.subplot(3, 3, 3 * i + 1)
    plt.xlabel(f"motion blur + Gaussian noise σ={sigmas[i]}")
    plt.imshow(img_corrupted,cmap='gray')

    # inverse filter
    plt.subplot(3,3,3*i+2)
    plt.xlabel("inverse filter result")
    plt.imshow(inverse_filter(img_corrupted, kernel),cmap='gray')

    # wiener filter
    plt.subplot(3,3,3*i+3)
    plt.xlabel(f"Wiener filter(k={ks[i]})")
    plt.imshow(wiener_filter(img_corrupted, kernel, ks[i]),cmap='gray')
plt.show()