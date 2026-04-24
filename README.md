# Stock-price-forecasting-using-LSTM-and-WRDS-data
A time-series prediction project that uses LSTM neural networks to forecast stock closing prices. Historical stock data is obtained from the WRDS CRSP database, followed by data preprocessing, sequence construction, model training, and evaluation with metrics including MSE and R².
## Part 1 Project Overview
This project builds an LSTM-based time-series forecasting model to predict Tesla (TSLA) stock prices using historical market data retrieved from the WRDS (Wharton Research Data Services) database. The workflow includes data acquisition, preprocessing, model training, and performance evaluation, demonstrating how recurrent neural networks can capture sequential patterns in financial time series.
## Part 2 Core code
```python
import wrds
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings('ignore')
ticker = input("Enter stock ticker (e.g. AAPL, MSFT, TSLA): ").strip().upper()
start_date = '2020-01-01'
time_steps = 60
train_ratio = 0.8
epochs = 25
batch_size = 32


print("Connecting to WRDS...")
db = wrds.Connection()
query = f"""
SELECT
    d.date,
    d.openprc,
    d.askhi,
    d.bidlo,
    d.prc,
    d.vol,
    d.ret
FROM crsp.dsf d
JOIN crsp.stocknames s
    ON d.permco = s.permco
WHERE s.ticker = '{ticker}'
    AND d.date >= '{start_date}'
ORDER BY d.date
"""

print(f"Downloading data for {ticker}...")
df = db.raw_sql(query)
db.close()
print("WRDS connection closed.")

if len(df) == 0:
    raise ValueError("Error: No data retrieved. Check ticker or date range.")
else:
    print(f"\nData downloaded successfully: {len(df)} rows")
    print(df.head())

print("\n=== Data Preprocessing ===")
df = df.rename(columns={
    'openprc': 'open',
    'askhi': 'high',
    'bidlo': 'low',
    'prc': 'close'
})

for col in ['open', 'high', 'low', 'close']:
    df[col] = df[col].abs()

df = df.dropna()
print(f"Rows after dropping NaN: {len(df)}")

df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df[['close']])

print("Preprocessing finished. Columns:", df.columns.tolist())

def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i-time_steps:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)
train_size = int(len(df_scaled) * train_ratio)
train_data = df_scaled[:train_size]
test_data = df_scaled[train_size:]

X_train, y_train = create_sequences(train_data, time_steps)
X_test, y_test = create_sequences(test_data, time_steps)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print(f"\nTrain shape: X={X_train.shape}, y={y_train.shape}")
print(f"Test shape: X={X_test.shape}, y={y_test.shape}")

print("\n=== Building LSTM Model ===")
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=25),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()
print("\n=== Training Model ===")
history = model.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_test, y_test),
    verbose=1
)

print("\n=== Prediction & Evaluation ===")
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = scaler.inverse_transform(train_predict)
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
test_predict = scaler.inverse_transform(test_predict)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

train_mse = mean_squared_error(y_train_actual, train_predict)
train_r2 = r2_score(y_train_actual, train_predict)
test_mse = mean_squared_error(y_test_actual, test_predict)
test_r2 = r2_score(y_test_actual, test_predict)

print(f"Train MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
print(f"Test MSE: {test_mse:.4f}, R²: {test_r2:.4f}")

plt.figure(figsize=(16, 8))
plt.plot(df.index, scaler.inverse_transform(df_scaled), label='Original Price', color='blue')
train_dates = df.index[time_steps:train_size]
plt.plot(train_dates, train_predict, label='Train Predictions', color='green')
test_dates = df.index[train_size + time_steps:]
plt.plot(test_dates, test_predict, label='Test Predictions', color='red')
plt.title(f'{ticker} Stock Price Prediction using LSTM')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.show()
```




## Part 3 Environment & Dependencies
This project is developed in Python using the following libraries:

**Pip install wrds pandas numpy matplotlib scikit-learn tensorflow**

- Python 3.8+(tested with Anaconda)

- TensorFlow 2.x(note:GPU acceleration is not supported for native Windows in TensorFlow ≥2.11,use WSL2 or TensorFlow-DirectML if needed)

- WRDS account required for data access
## Part 4 Data Source & Preprocessing
### 1. Data Acquisition
- Source: WRDS CRSP Daily Stock File(crsp.dsf)

- Target Ticker: TSLA

- Date Range: 2020-01-01 to latest available

- Features Retrieved: open, high , low , vol , ret
### 2. Prepocessing Steps
- Dropped missing values(NaN)

- Applied absolute value transformation to price colums (to handle potential negative values in raw CRSP data)

- Scaled the close price to [0,1] using MinMaxScaler for stable LSTM training

- Created sequences with a  time_steps=60 window (using the past 60 days to predict the next day's closing price)

- Split into training (80%) and test (20%) sets
## Part 5 LSTM Model Architecture
The model is a sequential LSTM neural network designed for time-series regression:

(<img width="1638" height="892" alt="360截图20260422150054395" src="https://github.com/user-attachments/assets/ee412fa4-6d99-4211-89c9-23f8be1dfa5e" />
)

**Input**: Sequences of shape (samples, 60, 1) (60-day closing prices)

**Architecture**:
  
  - 2 stacked LSTM layers with 50 units each

  - Dropout layers(rate=0.2)for regularization

  - Dense layers to map LSTM outputs to the predicted closing price

**Training**: Optimizer=Adam, Loss funtion=Mean Squared Error (MSE), Epochs=25, Batch size=32
## Part 6 Model Performance
### 1. Prediction vs. Actual Price
The model's predictions(green:training, red:test) closely follow the blue line of actual TSLA prices across the entire time range, including periods of high volatility(e.g., the 2020-2021 rally, 2022 correction, and post-2023 recovery).
### 2. Quantitative Metrics

|Metric|Training Set|Test Set|
|---|---|---|
|MSE|3525.49|245.29|
|R² Score|0.9738|0.9558|

- The high **R² Score(≈0.95-0.97)** indicate the model expains over 95% of the variance in TSLA’s closing prices.

- The test set MSE is significantly lower than the training set MSE, which aligns with the validation loss curve and suggests no severe overfitting.

<img width="687" height="285" alt="7" src="https://github.com/user-attachments/assets/b09ad4d2-cb4d-49fb-b751-f1fb3377b205" />
<img width="834" height="416" alt="6" src="https://github.com/user-attachments/assets/d5b6b255-58c5-4750-b5d8-de9cb9721bbc" />

### 3. Training & Validation Loss

The loss plot shows:

- Training loss (blue) decreases steadily in the first 10 epochs and stabilizes afterward.

- Validation loss (orange) remains consistently low throughout training, with no upward trend, confirming the model generalizes well to unseen data.
## Part 7 Key Notes & Limitations
### 1. Fiancial Forecasting Limitations:
- This model uses only historical price data and does not account for market news, macroeconomic factors, or company fundamentals.

- Past performance is not indicative of future results, especially for volatile stocks like TSLA.
### 2. Technical Notes:
- You will need a valid WRDS account to run the full data retrieval code.

- On native Windows, TensorFlow ≥2.11 does not support GPU acceleration. If training is slow, consider using WSL2 or TensorFlow-DirectML.   
### 3. Hyperparameter Tuning:
- Adjust time_steps, train_ratio, epochs, or LSTM units to optimize performance for different stocks.
## Part 8 How to Run
1. Install dependencies and set up your WRDS account.

2. Run the notebook/script

3. Enter your desired stock ticker (e.g., TSLA, AAPL) when prompted.

4. View the generated prediction plot and loss curve.
## Part 9 AI Disclosure

This README and the accompanying code were refined with the assistance of Doubao AI (access date: 2026-04-20). The AI was used for:

- Structuring the project documentation

- Summarizing model architecture and results

- Formatting the README for clarity

All code implementation, data analysis, and interpretation of results are the original work of the author.









