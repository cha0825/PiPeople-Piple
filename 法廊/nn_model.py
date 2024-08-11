# import model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd

# load trademark csv 
data = pd.read_csv('trademark.csv')

# select for x = ssim, hsv; y for similarity label
X = data[['ssim', 'hsv']]
y = data['sim_label']

# split train dataset and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# construct neural network model
model = Sequential()
# use relu to optimization
model.add(Dense(64, input_dim=2, activation='relu'))
model.add(Dense(32, activation='relu'))
# use linear sigmoid function 
model.add(Dense(1, activation='linear'))  

# compile
model.compile(optimizer='adam', loss='mean_squared_error')

# training
model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test))

# evaluate
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# 使用模型進行預測
predictions = model.predict(X_test)
print(predictions)
