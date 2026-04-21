# Stock-price-forecasting-using-LSTM-and-WRDS-data
A time-series prediction project that uses LSTM neural networks to forecast stock closing prices. Historical stock data is obtained from the WRDS CRSP database, followed by data preprocessing, sequence construction, model training, and evaluation with metrics including MSE and R².
## Part 1 Project Overview
This project builds an LSTM-based time-series forecasting model to predict Tesla (TSLA) stock prices using historical market data retrieved from the WRDS (Wharton Research Data Services) database. The workflow includes data acquisition, preprocessing, model training, and performance evaluation, demonstrating how recurrent neural networks can capture sequential patterns in financial time series.
## Part 2 Environment & Dependencies
This project is developed in Python using the following libraries:
### 2.1
Pip install wrds pandas numpy matplotlib scikit-learn tensorflow
### 2.2
1.Python 3.8+(tested with Anaconda)

2.TensorFlow 2.x(note:GPU acceleration is not supported for native Windows in TensorFlow ≥2.11,use WSL2 or TensorFlow-DirectML if needed)

3.WRDS account required for data access
## Part 3 Data Source & Preprocessing
### 3.1 Data Acquisition
1.Source: WRDS CRSP Daily Stock File(crsp.dsf)

2.Target Ticker:

3.Date Range: 2020-01-01 to latest available

4.Features Retrieved: open, high , low , vol , ret
### 3.2 Prepocessing Steps
1.Dropped missing values(NaN)

2.Applied absolute value transformation to price colums (to handle potential negative values in raw CRSP data)

3.Scaled the close price to [0,1] using MinMaxScaler for stable LSTM training

4.Created sequences with a  time_steps=60 window (using the past 60 days to predict the next day's closing price)

5.Split into training (80%) and test (20%) sets




