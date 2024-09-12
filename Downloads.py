# Hello there! 😊 
# If you're looking to gather data to train your model, this code will help you do just that.
# This script fetches data from Binance Futures, whether you're using the mainnet or the testnet.
# Before running the code, make sure to install the necessary packages with the following command:
# pip install python-binance pandas numpy

# If you encounter any issues or have questions, feel free to reach out to me on Telegram: https://t.me/AlrzA_2003


from binance.client import Client
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from binance.client import Client
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

with open("credentials.txt", "r") as file:  # load API Key and Secret from "credentials.txt" file.
    content = file.read()

lines = content.splitlines()
creds = {}
for line in lines:
    if "=" in line:
        key, value = line.split("=")
        creds[key.strip()] = value.strip()


api_key = creds["API_KEY"]
secret = creds["SECRET"]

timeframe = "15m" # Change the timeframe here.
trading_pair = "LINKUSDT" # Choose your pair here.
# Replace "LINKUSDT" and "15m" with any currency pair and timeframe you want.

# now = datetime.utcnow()
# start = str(now - timedelta(days = 60))
if __name__ == "__main__":
    client = Client(api_key = api_key, api_secret = secret, tld = "com", testnet = False) 
            # Set testnet to "True" if you're currently working with Binance Future Testnet.
    
    start = str(pd.to_datetime(client._get_earliest_valid_timestamp(trading_pair, timeframe), unit = "ms")) 
            
    
    
    bars = client.futures_historical_klines(symbol = trading_pair, interval = timeframe,
                                            start_str = start, end_str = None, limit = 1000)
    df = pd.DataFrame(bars)
    df["Date"] = pd.to_datetime(df.iloc[:,0], unit = "ms")
    df.columns = ["Open Time", "Open", "High", "Low", "Close", "Volume",
                    "Clos Time", "Quote Asset Volume", "Number of Trades",
                    "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore", "Date"]
    df = df[["Date", "Open", "High", "Low", "Close"]].copy()
    df.set_index("Date", inplace = True)
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors = "coerce")
    df.to_csv("{}_{}.csv".format(trading_pair, timeframe))
    print("Done!")

