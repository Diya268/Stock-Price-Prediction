
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import pandas as pd
from keras.models import load_model
import streamlit as st
import yfinance as yf

# ticker_symbols= ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'FB', 'TSLA', 'NFLX', 'NVDA', 'BABA', 'JNJ']
ticker_symbols = [
    'AAPL', # Apple Inc.
    'MSFT', # Microsoft Corporation
    'GOOGL', # Alphabet Inc.
    'AMZN', # Amazon.com Inc.
    'FB',    # Facebook, Inc.
    'TSLA', # Tesla, Inc.
    'NFLX', # Netflix, Inc.
    'NVDA', # NVIDIA Corporation
    'BABA', # Alibaba Group Holding Limited
    'JNJ',   # Johnson & Johnson
    'JPM',   # JPMorgan Chase & Co.
    'V',     # Visa Inc.
    'MA',    # Mastercard Incorporated
    'BAC',   # Bank of America Corporation
    'WMT',   # Walmart Inc.
    'HD',    # Home Depot, Inc.
    'DIS',   # The Walt Disney Company
    'XOM',   # Exxon Mobil Corporation
    'CVX',   # Chevron Corporation
    'VZ',    # Verizon Communications Inc.
    'KO',    # Coca-Cola Company (The)
    'PEP'    # PepsiCo, Inc.
]
start = "2015-01-01"
end = "2023-12-31"
st.title("Stock Price Prediction")
user_input = st.selectbox("Select a Stock Ticker", ticker_symbols)
# user_input = st.text_input("Enter Stock Ticker", "AAPL")
df = yf.download(user_input, start, end)

# Describing Data
st.subheader("Data from 2015 - 2023")
st.write(df.describe())


st.subheader("Closing Price vs Time chart")
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig)


st.subheader("Closing Price vs Time chart with 100MA & 200MA")
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, "r")
plt.plot(ma200, "g")
plt.plot(df.Close, "b")
st.pyplot(fig)



data_training = pd.DataFrame(df["Close"][0 : int(len(df) * 0.70)])
data_testing = pd.DataFrame(df["Close"][int(len(df) * 0.70) : int(len(df))])

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1), copy=True, clip=False)
data_training_array = scaler.fit_transform(data_training)



x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i - 100 : i])
    y_train.append(data_training_array[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)



model = load_model("keras_model.h5")

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100 : i])
    y_test.append(input_data[i, 0])


x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_
scale_factor = 1 / scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor



st.subheader("Predictions vs Original")
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, "b", label="Original Price")
plt.plot(y_predicted, "r", label="Predicted Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend(loc='best')
st.pyplot(fig2)