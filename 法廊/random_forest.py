from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib  # for saving the model

def train_rf_model(dataframe):
    # Check and clean NaN or inf values
    dataframe = dataframe.dropna()
    dataframe = dataframe.replace([np.inf, -np.inf], np.nan).dropna()

    # Define features and target
    X = dataframe[['ssimScore', 'hsvScore', 'cnnScore']].values
    y = dataframe['averageScore'].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the feature data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')

    average_loss = np.mean(np.abs(predictions - y_test))
    print(f"Average Loss: {average_loss}")

    # Optionally print first few predictions vs actual values
    for i in range(min(5, len(y_test))):  # Print first 5 predictions
        print(f"Predicted: {predictions[i]}, Actual: {y_test[i]}")

    # Save the model to a file
    joblib.dump(model, 'random_forest_model.pkl')
    print("Model saved as 'random_forest_model.pkl'")

    return model
