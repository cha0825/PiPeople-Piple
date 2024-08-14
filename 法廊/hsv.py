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

def concatenate_histograms(histograms):
    # Concatenate histograms (Hue, Saturation, Value) into one vector
    hist_h, hist_s, hist_v = histograms
    concatenated_hist = np.concatenate([hist_h, hist_s, hist_v])
    return concatenated_hist

def split_image_into_grid(image, grid_size=3):
    # Get image dimensions
    height, width = image.shape[:2]
    # Calculate the size of each grid cell
    cell_height = height // grid_size
    cell_width = width // grid_size
    cells = []
    # Extract each grid cell
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
    # Split the image into 3x3 grid cells
    cells = split_image_into_grid(image)
    # Compute histograms for each cell
    histograms = [compute_histogram(cell) for cell in cells]
    # Concatenate histograms for each cell into a single vector
    concatenated_histograms = [concatenate_histograms(histogram) for histogram in histograms]
    return concatenated_histograms

def compare_image_regions(image_path1, image_path2):
    # Load and convert images to HSV
    hsv_image1 = load_and_convert_to_hsv(image_path1)
    hsv_image2 = load_and_convert_to_hsv(image_path2)
    
    # Compute concatenated histograms for each region
    concatenated_histograms1 = compute_concatenated_histograms(hsv_image1)
    concatenated_histograms2 = compute_concatenated_histograms(hsv_image2)
    
    # Calculate similarities for each region
    similarities = []
    for hist1, hist2 in zip(concatenated_histograms1, concatenated_histograms2):
        sim = cosine_similarity(hist1.reshape(1, -1), hist2.reshape(1, -1))[0][0]
        similarities.append(sim)
    
    # Convert similarity scores to percentages
    similarity_percentages = [sim * 100 for sim in similarities]
    
    # Calculate the average similarity percentage for all regions
    average_similarity = np.mean(similarity_percentages)
    
    return similarity_percentages, average_similarity

def compare_entire_images(image_path1, image_path2):
    # Load and convert images to HSV
    hsv_image1 = load_and_convert_to_hsv(image_path1)
    hsv_image2 = load_and_convert_to_hsv(image_path2)
    
    # Compute concatenated histograms for the entire images
    concatenated_hist1 = concatenate_histograms(compute_histogram(hsv_image1))
    concatenated_hist2 = concatenate_histograms(compute_histogram(hsv_image2))
    
    # Compute cosine similarity for the entire images
    final_similarity = cosine_similarity(concatenated_hist1.reshape(1, -1), concatenated_hist2.reshape(1, -1))[0][0]
    
    # Convert similarity score to percentage
    similarity_percentage = final_similarity * 100
    return similarity_percentage

def plot_image(image_path, ax, title):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax.imshow(image)
    ax.set_title(title)
    ax.axis('off')

def plot_histograms(histograms, ax, title):
    bins = np.arange(256)
    colors = ['b', 'g', 'r']
    labels = ['Hue', 'Saturation', 'Value']
    
    for i, hist in enumerate(histograms):
        for j, color in enumerate(colors):
            ax.plot(bins, hist[j], color=color, label=f'{labels[j]} - Cell {i+1}' if i == 0 else "")
    ax.set_title(title)
    ax.set_xlabel('Bins')
    ax.set_ylabel('Frequency')
    ax.legend()

def main():
    # Paths to the images
    image_path1 = "1.jpg"
    image_path2 = "pic118.jpg"

    # Compare the images and get the similarity scores for each region
    similarity_scores, average_similarity = compare_image_regions(image_path1, image_path2)
    # Compare the entire images and get the similarity score
    overall_similarity_score = compare_entire_images(image_path1, image_path2)
    
    # Plot the images and histograms
    fig, axes = plt.subplots(4, 3, figsize=(18, 12))
    
    plot_image(image_path1, axes[0, 0], "Image 1")
    plot_image(image_path2, axes[0, 1], "Image 2")
    
    hsv_image1 = load_and_convert_to_hsv(image_path1)
    hsv_image2 = load_and_convert_to_hsv(image_path2)
    
    cells1 = split_image_into_grid(hsv_image1)
    cells2 = split_image_into_grid(hsv_image2)
    
    histograms1 = [compute_histogram(cell) for cell in cells1]
    histograms2 = [compute_histogram(cell) for cell in cells2]
    
    for i in range(len(histograms1)):
        plot_histograms([histograms1[i], histograms2[i]], axes[(i//3) + 1, i % 3], f'HSV Histogram - Region {i+1}')
    
    # Add the similarity scores below the histograms
    similarity_text = f"Region Similarity Scores: {', '.join(f'Region {i+1}: {score:.2f}%' for i, score in enumerate(similarity_scores))}"
    fig.text(0.5, 0.01, similarity_text, ha='center', va='center', fontsize=12, fontweight='bold', transform=fig.transFigure)
    
    # Add the overall similarity score at the bottom
    fig.text(0.5, 0.03, f"Overall Similarity: {overall_similarity_score:.2f}%", ha='center', va='center', fontsize=14, fontweight='bold', transform=fig.transFigure)
    
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig("figure1.png")
    plt.show()
    

if __name__ == "__main__":
    main()
