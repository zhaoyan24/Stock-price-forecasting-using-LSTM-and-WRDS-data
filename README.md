# Stock-price-forecasting-using-LSTM-and-WRDS-data
A time-series prediction project that uses LSTM neural networks to forecast stock closing prices. Historical stock data is obtained from the WRDS CRSP database, followed by data preprocessing, sequence construction, model training, and evaluation with metrics including MSE and R².
## Part 1 Project Overview
This project builds an LSTM-based time-series forecasting model to predict Tesla (TSLA) stock prices using historical market data retrieved from the WRDS (Wharton Research Data Services) database. The workflow includes data acquisition, preprocessing, model training, and performance evaluation, demonstrating how recurrent neural networks can capture sequential patterns in financial time series.
## Part 2 Environment & Dependencies
This project is developed in Python using the following libraries:

**Pip install wrds pandas numpy matplotlib scikit-learn tensorflow**

- Python 3.8+(tested with Anaconda)

- TensorFlow 2.x(note:GPU acceleration is not supported for native Windows in TensorFlow ≥2.11,use WSL2 or TensorFlow-DirectML if needed)

- WRDS account required for data access
## Part 3 Data Source & Preprocessing
### 1. Data Acquisition
- Source: WRDS CRSP Daily Stock File(crsp.dsf)

- Target Ticker:

- Date Range: 2020-01-01 to latest available

- Features Retrieved: open, high , low , vol , ret
### 2. Prepocessing Steps
- Dropped missing values(NaN)

- Applied absolute value transformation to price colums (to handle potential negative values in raw CRSP data)

- Scaled the close price to [0,1] using MinMaxScaler for stable LSTM training

- Created sequences with a  time_steps=60 window (using the past 60 days to predict the next day's closing price)

- Split into training (80%) and test (20%) sets
## Part 4 LSTM Model Architecture
The model is a sequential LSTM neural network designed for time-series regression:

(<img width="1638" height="892" alt="360截图20260422150054395" src="https://github.com/user-attachments/assets/ee412fa4-6d99-4211-89c9-23f8be1dfa5e" />
)

**Input**: Sequences of shape (samples, 60, 1) (60-day closing prices)

**Architecture**:
  
  - 2 stacked LSTM layers with 50 units each

  - Dropout layers(rate=0.2)for regularization

  - Dense layers to map LSTM outputs to the predicted closing price

**Training**: Optimizer=Adam, Loss funtion=Mean Squared Error (MSE), Epochs=25, Batch size=32
## Part 5 Model Performance
### 1. Prediction vs. Actual Price
The model's predictions(green:training, red:test) closely follow the blue line of actual TSLA prices across the entire time range, including periods of high volatility(e.g., the 2020-2021 rally, 2022 correction, and post-2023 recovery).
### 2. Quantitative Metrics








