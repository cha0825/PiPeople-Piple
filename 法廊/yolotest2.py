# -*- coding: utf-8 -*-
import torch
from PIL import Image, ImageFilter
import cv2
import numpy as np
import os
from ultralytics import YOLO

# 使用GPU加速
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用裝置: {device}")

# 載入 YOLOv10
model = YOLO('/content/drive/MyDrive/好快/yolov10x.pt').to(device)  # 自行下載Yolo模型並指定路徑

# 載入圖片
img_path1 = '1.jpg'  # 請修改為實際的圖片路徑
img_path2 = 'pic1.jpg'
img_cv2 = cv2.imread(img_path1)
img_cv2_2 = cv2.imread(img_path2)

# 圖片大小處理
scale_percent = 200  # 圖片放大100%
width = int(img_cv2.shape[1] * scale_percent / 300)
height = int(img_cv2.shape[0] * scale_percent / 300)
dim = (width, height)
img_cv2 = cv2.resize(img_cv2, dim, interpolation=cv2.INTER_LINEAR)

width2 = int(img_cv2_2.shape[1] * scale_percent / 300)
height2 = int(img_cv2_2.shape[0] * scale_percent / 300)
dim2 = (width2, height2)
img_cv2_2 = cv2.resize(img_cv2_2, dim2, interpolation=cv2.INTER_LINEAR)

# 圖片轉換為 PIL 圖片，然後轉換為 RGB 格式
img_pil = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
img_pil_2 = Image.fromarray(cv2.cvtColor(img_cv2_2, cv2.COLOR_BGR2RGB))

# 進行檢測
results = model(img_pil)
results2 = model(img_pil_2)

# 獲取檢測結果
detections = results[0].boxes.data.cpu().numpy()  # 0 代表第一張圖片
detections_2 = results2[0].boxes.data.cpu().numpy()

# 用來計算物件類別數量
object_counts = {}
object_counts_2 = {}

# 處理第一張圖片中的檢測到的物件
for idx, (*box, conf, cls) in enumerate(detections):
    class_name = model.names[int(cls)]
    if class_name in object_counts:
        object_counts[class_name] += 1
    else:
        object_counts[class_name] = 1

# 處理第二張圖片中的檢測到的物件
for idx, (*box, conf, cls) in enumerate(detections_2):
    class_name = model.names[int(cls)]
    if class_name in object_counts_2:
        object_counts_2[class_name] += 1
    else:
        object_counts_2[class_name] = 1

# 找出每張圖片中出現最多的物件及其數量
max_count_1 = max(object_counts.values(), default=0)
max_objects_1 = [obj for obj, count in object_counts.items() if count == max_count_1]

max_count_2 = max(object_counts_2.values(), default=0)
max_objects_2 = [obj for obj, count in object_counts_2.items() if count == max_count_2]

# 輸出物件數量
print("第一張圖片物件數量:")
for object_name, count in object_counts.items():
    print(f"物件 {object_name} 的數量: {count}")
print(f"最多物件數量: {max_count_1}，物件類型: {', '.join(max_objects_1)}")

print("\n第二張圖片物件數量:")
for object_name, count in object_counts_2.items():
    print(f"物件 {object_name} 的數量: {count}")
print(f"最多物件數量: {max_count_2}，物件類型: {', '.join(max_objects_2)}")

# 輸出相同標籤的物件
print("\n相同標籤的物件:")
common_objects = set(object_counts.keys()) & set(object_counts_2.keys())
for object_name in common_objects:
    print(f"物件 {object_name} - 第一張圖片: {object_counts[object_name]}，第二張圖片: {object_counts_2[object_name]}")

# (可選) 顯示結果圖片
# from matplotlib import pyplot as plt
# plt.imshow(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.show()
