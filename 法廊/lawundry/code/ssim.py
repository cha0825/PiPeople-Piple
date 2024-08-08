import numpy as np
import cv2
from scipy.signal import convolve2d
import os
import sys

#樣式處裡
def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

#過濾
def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)

#ssim
def compute_ssim(im1, im2, k1=0.01, k2=0.04, win_size=11, L=255):
    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    window = matlab_style_gauss2D(shape=(win_size, win_size), sigma=0.5)
    window = window / np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1 * im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2 * im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1 * im2, window, 'valid') - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigmal2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return np.mean(np.mean(ssim_map))

dir_path = os.path.dirname(os.path.abspath(__file__))
image_path1 = os.path.join(dir_path, sys.argv[1])
image_path2 = os.path.join(dir_path, sys.argv[2])

img1 = cv2.imread(image_path1)
img2 = cv2.imread(image_path2)

im1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
im2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

im1 = cv2.resize(im1, (520, 520))
im2 = cv2.resize(im2, (520, 520))

similarity = compute_ssim(im1, im2) * 100
print(f'{similarity:.2f}%')