#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from Indicators import Indicators
from stock_indicators.indicators.common.enums import EndType
from stock_indicators.indicators.common.quote import Quote
from Labels import get_zig_zag
from sklearn.preprocessing import MinMaxScaler
from joblib import dump
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# the LINKUSDT_15m.csv file Is 15m candles of LINK/USDT currency pair and it has Date, Open, High, Low, Close columns
data = pd.read_csv("LINKUSDT_15m.csv", parse_dates = ["Date"], index_col = "Date")
ind = Indicators(data)
ind.all_ind()
pca = ind.add_pca(1)
data = ind.data.copy()
data["Feature_1"] = pca
df = data[["Open", "High", "Low", "Close"]].copy()
def zigzag(df, source = EndType.HIGH_LOW, pct = 5):
    # Assuming you have a pandas DataFrame named `df` with columns 'Open', 'High', 'Low', and 'Close'
    quotes = [Quote(date=row.Index, open=row.Open, high=row.High, low=row.Low, close=row.Close) for row in df.itertuples()]

    # Calculate the Zig Zag indicator
    zig_zag_results = get_zig_zag(quotes, end_type = source, percent_change = pct)

    # Access the Zig Zag results
    trend = []
    for result in zig_zag_results:
        if result.point_type == "H":
            trend.append(0)
        elif result.point_type == "L":
            trend.append(1)
        else:
            trend.append(None)
    s = pd.Series(trend, index = df.index)
    s.ffill(inplace = True)
    return s.shift(-1)
data["Labels"] = zigzag(df = df, source = EndType.HIGH_LOW, pct = 2)
lags = []
for i in range(1, 13):
    lag = f"label_lag_{i}"
    data[lag] = data["Labels"].shift(i)
    lags.append(lag)
dropping_cols = ['EMA', 'Open', 'High', 'Close', 'Low']
data.drop(dropping_cols, axis = 1, inplace = True)
data.dropna(inplace = True)
X = data.drop("Labels", axis = 1).values
y = data["Labels"].values
scaler = MinMaxScaler()
scaler.fit(X)
dump(scaler, "scaler.joblib")
X = scaler.transform(X)
early_stop = EarlyStopping(monitor = "loss", mode = "min", verbose = 1, patience = 25)

model = Sequential()
model.add(Dense(52, input_dim = 51, activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(26, activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(14, activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(x = X, y = y, epochs = 100, callbacks = [early_stop], batch_size = 16)
model.save("model.h5")

