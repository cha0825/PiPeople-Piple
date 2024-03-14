from matplotlib import pyplot as plt
import argparse
import cv2

 
chans = cv2.split(image)  # 將讀取的image檔split為channels
colors = (“b", “g", “r")   # 定義一colors數組，注意openCV split後的channels排序為BGR

plt.figure()
plt.title(“‘Flattened’ Color Histogram")
plt.xlabel(“Bins")
plt.ylabel(“# of Pixels")

 

# 依次取出chans及colors的B,G,R值繪製直方圖

for (chan, color) in zip(chans, colors):
       
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        plt.plot(hist, color = color)
        plt.xlim([0, 256])