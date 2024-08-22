# hsv 切36格

import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import sys
import os

def load_and_convert_to_hsv(image_path):
    image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv_image

def compute_histogram(image, bins=256):
    hist_h = cv2.calcHist([image], [0], None, [bins], [0, 256])
    hist_s = cv2.calcHist([image], [1], None, [bins], [0, 256])
    hist_v = cv2.calcHist([image], [2], None, [bins], [0, 256])
    hist_h = cv2.normalize(hist_h, hist_h).flatten()
    hist_s = cv2.normalize(hist_s, hist_s).flatten()
    hist_v = cv2.normalize(hist_v, hist_v).flatten()
    return hist_h, hist_s, hist_v

def concatenate_histograms(histograms):
    hist_h, hist_s, hist_v = histograms
    concatenated_hist = np.concatenate([hist_h, hist_s, hist_v])
    return concatenated_hist

def split_image_into_grid(image, grid_size=6):
    height, width = image.shape[:2]
    cell_height = height // grid_size
    cell_width = width // grid_size
    cells = []
    for i in range(grid_size):
        for j in range(grid_size):
            y1 = i * cell_height
            y2 = (i + 1) * cell_height
            x1 = j * cell_width
            x2 = (j + 1) * cell_width
            cell = image[y1:y2, x1:x2]
            cells.append(cell)
    return cells

def compute_concatenated_histograms(image):
    cells = split_image_into_grid(image)
    histograms = [compute_histogram(cell) for cell in cells]
    concatenated_histograms = [concatenate_histograms(histogram) for histogram in histograms]
    return concatenated_histograms

def compare_image_regions(image_path1, image_path2):
    hsv_image1 = load_and_convert_to_hsv(image_path1)
    hsv_image2 = load_and_convert_to_hsv(image_path2)
    
    concatenated_histograms1 = compute_concatenated_histograms(hsv_image1)
    concatenated_histograms2 = compute_concatenated_histograms(hsv_image2)
    
    similarities = []
    for hist1, hist2 in zip(concatenated_histograms1, concatenated_histograms2):
        sim = cosine_similarity(hist1.reshape(1, -1), hist2.reshape(1, -1))[0][0]
        similarities.append(sim)
    
    similarity_percentages = [sim * 100 for sim in similarities]
    average_similarity = np.mean(similarity_percentages)
    
    return similarity_percentages, average_similarity

def compare_entire_images(image_path1, image_path2):
    hsv_image1 = load_and_convert_to_hsv(image_path1)
    hsv_image2 = load_and_convert_to_hsv(image_path2)
    
    concatenated_hist1 = concatenate_histograms(compute_histogram(hsv_image1))
    concatenated_hist2 = concatenate_histograms(compute_histogram(hsv_image2))
    
    final_similarity = cosine_similarity(concatenated_hist1.reshape(1, -1), concatenated_hist2.reshape(1, -1))[0][0]
    
    similarity_percentage = final_similarity * 100
    return similarity_percentage



def main(image_path1, image_path2):

    similarity_scores, average_similarity = compare_image_regions(image_path1, image_path2)
    
    print(f"{average_similarity:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_images.py <image1_path> <image2_path>")
        sys.exit(1)

    image_path1 = sys.argv[1]
    image_path2 = sys.argv[2]
    
    main(image_path1, image_path2)