import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import yfinance as yf

# Regression models
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

# Neural networks
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Load dataset from Yahoo Finance
ticker = 'SBIN.NS'  # State Bank of India NSE ticker
df = yf.download(ticker, start="2015-01-01", end="2024-01-01")
df['Date'] = df.index

# Preprocessing
df = df[['Date', 'Close']]  # Using only Date and Close Price
df['Close'] = df['Close'].fillna(method='ffill')

# Feature Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
df['Close_scaled'] = scaler.fit_transform(np.array(df['Close']).reshape(-1, 1))

# Splitting data into training and testing sets
train_size = int(len(df) * 0.8)
train_data, test_data = df[:train_size], df[train_size:]

# Creating train and test features
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X_train, y_train = create_dataset(train_data['Close_scaled'].values.reshape(-1, 1), time_step)
X_test, y_test = create_dataset(test_data['Close_scaled'].values.reshape(-1, 1), time_step)

# Reshaping data
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Evaluation metrics
def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return rmse, mae, mape

# Initialize models
models = {
    "Linear Regression": LinearRegression(),
    "SVR": SVR(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "XGBoost": XGBRegressor(),
    "AdaBoost": AdaBoostRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "KNN": KNeighborsRegressor()
}

# Train and evaluate each model
for name, model in models.items():
    # Reshape X_train and X_test for non-ANN models
    X_train_flat = X_train.reshape(X_train.shape[0], X_train.shape[1])
    X_test_flat = X_test.reshape(X_test.shape[0], X_test.shape[1])

    # Train the model
    model.fit(X_train_flat, y_train)

    # Predict
    predictions = model.predict(X_test_flat)
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))

    # Evaluate
    rmse, mae, mape = evaluate_model(test_data['Close'].values[:len(predictions)], predictions)
    print(f"{name} - RMSE: {rmse}, MAE: {mae}, MAPE: {mape}")

# ANN Model
ann_model = Sequential()
ann_model.add(Dense(50, input_shape=(time_step, 1), activation='relu'))
ann_model.add(Dense(25, activation='relu'))
ann_model.add(Dense(1))
ann_model.compile(optimizer='adam', loss='mean_squared_error')

# Train ANN model
ann_model.fit(X_train, y_train, epochs=20, batch_size=64, verbose=1)

# Predict with ANN
ann_predictions = ann_model.predict(X_test)
ann_predictions = scaler.inverse_transform(ann_predictions)

# Evaluate ANN
rmse, mae, mape = evaluate_model(test_data['Close'].values[:len(ann_predictions)], ann_predictions)
print(f"ANN Model - RMSE: {rmse}, MAE: {mae}, MAPE: {mape}")

# LSTM Model
lstm_model = Sequential()
lstm_model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
lstm_model.add(LSTM(50, return_sequences=False))
lstm_model.add(Dense(25))
lstm_model.add(Dense(1))

lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Train LSTM model
lstm_model.fit(X_train, y_train, epochs=20, batch_size=64, verbose=1)

# Predict with LSTM
lstm_predictions = lstm_model.predict(X_test)
lstm_predictions = scaler.inverse_transform(lstm_predictions)

# Evaluate LSTM
rmse, mae, mape = evaluate_model(test_data['Close'].values[:len(lstm_predictions)], lstm_predictions)
print(f"LSTM Model - RMSE: {rmse}, MAE: {mae}, MAPE: {mape}")

# Plot LSTM vs Actual
plt.plot(test_data['Date'][:len(lstm_predictions)], test_data['Close'].values[:len(lstm_predictions)], label='Actual Price')
plt.plot(test_data['Date'][:len(lstm_predictions)], lstm_predictions, label='LSTM Predicted Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('State Bank of India Stock Price Prediction')
plt.legend()
plt.show()
