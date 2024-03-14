import numpy as np
import cv2
from scipy.signal import convolve2d
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog




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

#大小裁切+拼接
def img_show(similarity, img1, img2, name1, name2):
    img1 = cv2.resize(img1, (520, 520))
    img2 = cv2.resize(img2, (520, 520))
    imgs = np.hstack([img1, img2])
    path = "{0}".format('{0}VS{1}相似指數{2}%.jpg'.format(name1, name2, round(similarity, 2)))
    cv2.imencode('.jpg', imgs)[1].tofile(path)
    return path

#pic要自己改成圖片名稱
name1 = "pic1.jpg"
name2 = "pic2.jpg"

img1_path = "pic1.jpg"
img2_path = "pic2.jpg"

img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

im1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
im2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

im1 = cv2.resize(im1, (520, 520))
im2 = cv2.resize(im2, (520, 520))

similarity = compute_ssim(im1, im2) * 100
similarity = compute_ssim(im1, im2) * 100
if similarity == 100:
    print("圖片相似度百分之百!")
    sys.exit()

comparetext = (img_show(similarity, img1 , img2, name1 , name2))


# 建立視窗
top = tk.Tk() 
top.title('對比') 
top.geometry('1200x700+200+100') 


# 讀取圖片
picture1 = ImageTk.PhotoImage(Image.open('pic1.jpg'))
picture2 = ImageTk.PhotoImage(Image.open('pic2.jpg'))



#按鈕函式
def compare_images():
    label_left_text.config(text= comparetext)

# 顯示圖片
label_left = tk.Label(top, height=560, width=480, bg='gray94', fg='blue', image=picture1) 
label_right = tk.Label(top, height=560, width=480, bg='gray94', fg='blue', image=picture2) 

label_left.grid(row=0, column=0, padx=10, pady=10)
label_right.grid(row=0, column=1, padx=10, pady=10)

# 建立文字label
label_left_text = tk.Label(top, text="", height=10, width=30)
label_left_text.grid(row=1, column=0, padx=10, pady=10)

# 建立比較按鈕
compare_button = tk.Button(top, text="比較圖片", command=compare_images)
compare_button.grid(row=1, column=1, padx=10, pady=10)

# 執行gui
top.mainloop()