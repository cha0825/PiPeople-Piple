import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

def load_and_convert_to_hsv(image_path):
    # Load the image
    image = cv2.imread(image_path)
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv_image

def compute_histogram(image, bins=256):
    # Compute histograms for each channel (Hue, Saturation, Value)
    hist_h = cv2.calcHist([image], [0], None, [bins], [0, 256])
    hist_s = cv2.calcHist([image], [1], None, [bins], [0, 256])
    hist_v = cv2.calcHist([image], [2], None, [bins], [0, 256])
    # Normalize the histograms
    hist_h = cv2.normalize(hist_h, hist_h).flatten()
    hist_s = cv2.normalize(hist_s, hist_s).flatten()
    hist_v = cv2.normalize(hist_v, hist_v).flatten()
    return hist_h, hist_s, hist_v

def cosine_similarity_between_histograms(hist1, hist2):
    # Compute cosine similarity between two histograms
    return cosine_similarity(hist1.reshape(1, -1), hist2.reshape(1, -1))[0][0]

def compare_images(image_path1, image_path2):
    # Load and convert images to HSV
    hsv_image1 = load_and_convert_to_hsv(image_path1)
    hsv_image2 = load_and_convert_to_hsv(image_path2)
    
    # Compute histograms for each image
    hist_h1, hist_s1, hist_v1 = compute_histogram(hsv_image1)
    hist_h2, hist_s2, hist_v2 = compute_histogram(hsv_image2)
    
    # Compute cosine similarities for each channel
    sim_h = cosine_similarity_between_histograms(hist_h1, hist_h2)
    sim_s = cosine_similarity_between_histograms(hist_s1, hist_s2)
    sim_v = cosine_similarity_between_histograms(hist_v1, hist_v2)
    
    # Average the cosine similarities for the final similarity score
    final_similarity = (sim_h + sim_s + sim_v) / 3.0
    
    # Convert similarity score to percentage
    similarity_percentage = final_similarity * 100
    return similarity_percentage

def plot_image(image_path, ax, title):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax.imshow(image)
    ax.set_title(title)
    ax.axis('off')

def plot_histogram(hist, ax, title):
    bins = np.arange(256)
    ax.plot(bins, hist[0], color='b', label='Hue')
    ax.plot(bins, hist[1], color='g', label='Saturation')
    ax.plot(bins, hist[2], color='r', label='Brightness')
    ax.set_title(title)
    ax.set_xlabel('Bins')
    ax.set_ylabel('Frequency')
    ax.legend()

def main():
    # Paths to the images
    image_path1 = "pic1.jpg"
    image_path2 = "pic2.jpg"

    # Compare the images and get the similarity score
    similarity_score = compare_images(image_path1, image_path2)
    
    # Plot the images and histograms
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    
    plot_image(image_path1, axes[0, 0], "Image 1")
    plot_image(image_path2, axes[0, 1], "Image 2")
    
    hsv_image1 = load_and_convert_to_hsv(image_path1)
    hsv_image2 = load_and_convert_to_hsv(image_path2)
    hist1 = compute_histogram(hsv_image1)
    hist2 = compute_histogram(hsv_image2)
    
    plot_histogram(hist1, axes[1, 0], "HSV Histogram - Image 1")
    plot_histogram(hist2, axes[1, 1], "HSV Histogram - Image 2")
    
    # Add the similarity score below the histograms
    fig.text(0.5, 0.01, f"Similarity: {similarity_score:.2f}%", ha='center', va='center', fontsize=20, fontweight='bold', transform=fig.transFigure)
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig("figure1.png")
    plt.show()
    

if __name__ == "__main__":
    main()
