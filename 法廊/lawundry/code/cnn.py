import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
# 為了去除跑程式碼過程會被顯示出來 
import cv2
import numpy as np
import argparse
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity

tf.get_logger().setLevel('ERROR')

def calculate_cnn_similarity(image1_path, image2_path):
    # Load VGG16 model pretrained on ImageNet
    base_model = VGG16(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
    
    # Read and preprocess images
    def preprocess_image(image_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))
        image = image.astype('float32')
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)  # Preprocess using VGG16's preprocess_input
        return image
    
    image1 = preprocess_image(image1_path)
    image2 = preprocess_image(image2_path)
    
    # Extract features from both images
    features1 = model.predict(image1, verbose=0)
    features2 = model.predict(image2, verbose=0)
    
    # Compute cosine similarity
    similarity = cosine_similarity(features1, features2)
    
    return similarity[0][0] * 100

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate CNN similarity between two images.")
    parser.add_argument("image1_path", type=str, help="Path to the first image")
    parser.add_argument("image2_path", type=str, help="Path to the second image")
    args = parser.parse_args()

    similarity_score = calculate_cnn_similarity(args.image1_path, args.image2_path)
    print(f"{similarity_score:.2f}%")
