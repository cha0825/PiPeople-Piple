import cv2
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity

def calculate_cnn_similarity(image1_path, image2_path):
    # Load VGG16 model pretrained on ImageNet
    base_model = VGG16(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
    
    # Read and preprocess images
    def preprocess_image(image_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (520, 520))
        image = image.astype('float32')
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)  # Preprocess using VGG16's preprocess_input
        return image
    
    image1 = preprocess_image(image1_path)
    image2 = preprocess_image(image2_path)
    
    # Extract features from both images
    features1 = model.predict(image1)
    features2 = model.predict(image2)
    
    # Compute cosine similarity
    similarity = cosine_similarity(features1, features2)
    
    return similarity[0][0]