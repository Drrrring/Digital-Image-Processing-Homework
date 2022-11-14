import cv2
import numpy as np

# (a) Original image a
image_a = cv2.imread('Resource/BodyScan.jpg')
cv2.imshow('Original', image_a)

# (b) Laplacian of a
kernel = np.array([[0, -1, 0],
                   [-1, 4, -1],
                   [0, -1, 0]])
image_b = cv2.filter2D(src=image_a, ddepth=-1, kernel=kernel)
# image_b = cv2.Laplacian(image_a, -1, ksize=3)
cv2.imshow('Laplacian', image_b)

# (c) add a and b
image_c = cv2.add(image_a, image_b)
cv2.imshow('Laplacian Scaling', image_c)

# (d) Sobel gradient of (a)
scale = 1
delta = 0
ddepth = cv2.CV_64F
grad_x = cv2.Sobel(image_a, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
grad_y = cv2.Sobel(image_a, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)
image_d = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
cv2.imshow('Sobel gradient', image_d)

# grad_xx = cv2.Sobel(image, ddepth, 1, 0, ksize=5, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
# grad_yy = cv2.Sobel(image, ddepth, 0, 1, ksize=5, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
# abs_grad_xx = cv2.convertScaleAbs(grad_xx)
# abs_grad_yy = cv2.convertScaleAbs(grad_yy)
# grad2 = cv2.addWeighted(abs_grad_xx, 0.5, abs_grad_yy, 0.5, 0)

# (e) Sobel image smoothed with a 5 × 5 box filter.
image_e = cv2.blur(image_d, (5, 5))
cv2.imshow('Sobel gradient2', image_e)

# (f) Mask image formed by the product of (b) and (e)
image_f = cv2.multiply(image_b, image_e)
cv2.imshow('mask', image_f)

# (g) Sharpened image obtained by the adding images (a) and (f)
image_g = cv2.add(image_a, image_f)
cv2.imshow('adding images (a) and (f)', image_g)

# (h) Final result obtained by applying a power law(gamma) transformation to (g).
image_h = np.array(255 * (image_g / 255) ** 0.5, dtype='uint8')
cv2.imshow('final', image_h)

cv2.waitKey()
cv2.destroyAllWindows()
