import pandas as pd
import os
import cv2
from skimage.metrics import structural_similarity as compute_ssim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def calculate_ssim(firstPicture, secondPicture):
    # Check if both files exist
    if not os.path.exists(firstPicture):
        print(f"Warning: {firstPicture} does not exist.")
        return None
    if not os.path.exists(secondPicture):
        print(f"Warning: {secondPicture} does not exist.")
        return None
    
    # Read the images
    img1 = cv2.imread(firstPicture)
    img2 = cv2.imread(secondPicture)
    
    # Check if images were loaded correctly
    if img1 is None:
        print(f"Error: Failed to load image {firstPicture}.")
        return None
    if img2 is None:
        print(f"Error: Failed to load image {secondPicture}.")
        return None

    # Convert images to grayscale
    im1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Resize images to a common size
    im1 = cv2.resize(im1, (520, 520))
    im2 = cv2.resize(im2, (520, 520))

    # Calculate SSIM
    similarity = compute_ssim(im1, im2) * 100
    return similarity

def calculate_hsv_similarity(image1_path, image2_path):
    # Read images
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    
    # Resize images to a common size
    im1 = cv2.resize(image1, (520, 520))
    im2 = cv2.resize(image2, (520, 520))
    
    # Convert images to HSV color space
    hsv_image1 = cv2.cvtColor(im1, cv2.COLOR_BGR2HSV)
    hsv_image2 = cv2.cvtColor(im2, cv2.COLOR_BGR2HSV)
    
    # Calculate histograms and normalize
    hist1 = cv2.calcHist([hsv_image1], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist2 = cv2.calcHist([hsv_image2], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    
    # Compute histogram similarity using correlation
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    return similarity

def read_and_preprocess():
    # read TradeMark.csv files as DataFrame
    df = pd.read_csv('../TradeMark.csv')
    # define the replace dictionary
    replace_dict = {
        "1 極為不相似": 1,
        "2": 2,
        "3": 3,
        "4 極為相似": 4
    }
    # replace the df by dict value
    df.replace(replace_dict, inplace=True)
    
    # Create a new DataFrame with selected columns
    new_df = pd.DataFrame()
    new_df['firstColumn'] = df['Unnamed: 1']
    new_df['secondColumn'] = df['Unnamed: 2']
    
    # Calculate SSIM score
    new_df['ssimScore'] = new_df.apply(
    lambda row: calculate_ssim('../imagesdata/' + row['firstColumn'], '../imagesdata/' + row['secondColumn']),
    axis=1
    )
    # Drop rows where 'ssimScore' is None (which is treated as NaN in Pandas)
    new_df.dropna(subset=['ssimScore'], inplace=True)
    
    # Calculate HSV score
    new_df['hsvScore'] = new_df.apply(
        lambda row: calculate_hsv_similarity('../imagesdata/' + row['firstColumn'], '../imagesdata/' + row['secondColumn']),
        axis=1
    )
    
    # Calculate the average score from columns 'Unnamed: 3' to 'Unnamed: 12'
    new_df['averageScore'] = df.loc[:, 'Unnamed: 3':'Unnamed: 12'].mean(axis=1)
    
    new_df.to_csv('test.csv')
    return new_df

def train_model(dataframe):
    # Define features and target
    X = dataframe[['ssimScore', 'hsvScore']].values
    y = dataframe['averageScore'].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the feature data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Define the model
    model = Sequential()

    # Add input layer and first hidden layer
    model.add(Dense(64, input_dim=2, activation='relu'))

    # Add second hidden layer
    model.add(Dense(32, activation='relu'))

    # Add output layer
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
    
    # Evaluate the model
    loss = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss}')
    
    # Make predictions
    predictions = model.predict(X_test)

    # Compare predictions with actual values
    for i in range(5):  # Print first 5 predictions
        print(f"Predicted: {predictions[i][0]}, Actual: {y_test[i]}")

    
    

df = read_and_preprocess()
train_model(df)



# # import model
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from sklearn.model_selection import train_test_split
# import pandas as pd

# # load trademark csv 
# data = pd.read_csv('trademark.csv')

# # select for x = ssim, hsv; y for similarity label
# X = data[['ssim', 'hsv']]
# y = data['sim_label']

# # split train dataset and test dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # construct neural network model
# model = Sequential()
# # use relu to optimization
# model.add(Dense(64, input_dim=2, activation='relu'))
# model.add(Dense(32, activation='relu'))
# # use linear sigmoid function 
# model.add(Dense(1, activation='linear'))  

# # compile
# model.compile(optimizer='adam', loss='mean_squared_error')

# # training
# model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test))

# # evaluate
# loss = model.evaluate(X_test, y_test)
# print(f'Test Loss: {loss}')

# # 使用模型進行預測
# predictions = model.predict(X_test)
# print(predictions)
