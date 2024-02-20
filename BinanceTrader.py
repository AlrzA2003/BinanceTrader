import ccxt
import pandas as pd
import numpy as np
import schedule
import pytz
from threading import Thread
from time import sleep
from datetime import datetime
from Indicators import Indicators
from joblib import load
from tensorflow.keras.models import load_model
from stock_indicators.indicators.common.enums import EndType
from stock_indicators.indicators.common.quote import Quote
from Labels import get_zig_zag

class BinanceTrader:
    """
    A class for algorithmic trading on Binance using a deep neural network model.

    Attributes
    ----------
    symbol : str
        The currency pair for algorithmic trading (e.g. "LINK/USDT").
    api_key : str
        The API key obtained from a Binance account.
    secret_key : str
        The secret key obtained from a Binance account.
    model_path : str
        The path to the saved deep neural network model on the local computer or server.
    scaler_path : str
        The path to the saved joblib scaler on the local computer or server.
    bar_length : str
        The timeframe of candles to fetch and trade (default is "15m").
    limit : int
        The number of candles to get per each try (default is 1000).
    leverage : int
        The leverage to trade with (default is 2).
    """
    
    def __init__(self, symbol, api_key, secret_key, testnet, model_path, scaler_path, bar_length = "15m", limit = 1000, leverage = 2):
        self.symbol = symbol
        self._api_key = api_key
        self._secret = secret_key
        self._testnet = testnet
        self._model = load_model(model_path)
        self._scaler = load(scaler_path)
        self.bar_length = bar_length
        self._limit = limit
        self.leverage = leverage
        self.position = 0
        self.data = None
        self.n = 0
        self.pnl = None
        self._notional = 1
        self.make_connection()
        self.fetch_balances()
        self.get_data()
        
    def __repr__(self):
        time = self.current_time()
        return f"{time} | BinanceTrader Class that uses DNN for predictions!"
    
    def current_time(self): # Fetching current time for print out (timezone = Iran)
        tz_Iran = pytz.timezone("Iran")
        current_time = datetime.now(tz_Iran)
        dis_time = current_time.strftime("%Y-%m-%d | %H:%M:%S")
        return dis_time
    
    def make_connection(self): # Make a connection to Binance.com
        binance = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}, 'timeout': 30000})
        binance.set_sandbox_mode(self._testnet) # For test mode, enable this option
        binance.apiKey = self._api_key
        binance.secret = self._secret
        binance.load_markets()
        binance.set_leverage(self.leverage, symbol = self.symbol)
        self.binance = binance
        
    def fetch_balances(self): # Account Balance
        ex = self.binance
        bal = ex.fetch_balance()["total"]["USDT"]
        price = ex.fetch_ticker(symbol = self.symbol)["last"]
        am = round(round(bal / price, 2) - round(1 / price, 2), 0)
        self._amount = am
        
    def get_data(self): # fetching the data from Binance using API
        ex = self.binance
        dates = []
        prices = []
        for item in ex.fetch_ohlcv(self.symbol, timeframe = self.bar_length, limit = self._limit):
            item = item[:-1]
            date, *pr = item
            dates.append(pd.to_datetime(date, unit = "ms"))
            prices.append(pr)
        df = pd.DataFrame(index = dates, data = prices, columns = ["Open", "High", "Low", "Close"])
        self.data = df
        
    def prepare_data(self): # Preparing Features and Labels
        data = self.data.copy()
        ind = Indicators(data)
        ind.all_ind()
        pca = ind.add_pca(1)
        df = ind.data.copy()
        df["Feature_1"] = pca
        df["Labels"] = self.zigzag(source = EndType.HIGH_LOW, pct = 2)
        for i in range(1, 13):
            lag = f"label_lag_{i}"
            df[lag] = df["Labels"].shift(i)
        dropping_cols = ['EMA', 'Open', 'High', 'Close', 'Low']
        df.drop(dropping_cols, axis = 1, inplace = True)
        self.data = df
        
    def zigzag(self, source = EndType.HIGH_LOW, pct = 5): # Creating Labels
        df = self.data[["Open", "High", "Low", "Close"]].copy()
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
    
    def prepare_model(self): # Preparing the stored model for prediction
        data = self.data.copy()
        model = self._model
        scaler = self._scaler
        X = data.iloc[-1].drop("Labels").values
        X = scaler.transform(X.reshape(1, len(data.columns) - 1))
        label = np.round(model.predict(X, verbose = None)[0, 0])
        if label == 1:
            self._label = 1
        elif label == 0:
            self._label = -1
        
        
    def strategy(self): # Opening LONG/SHORT positions based on predictions
        ex = self.binance
        if self._label == 1:
            if self.position == -1:
                order = ex.create_market_order(self.symbol, side = "buy", amount = self.units)
                self.print_status("close_s", order)
                self.position = 0
                sleep(1)
            if self.position == 0:
                flag = True
                for item in np.arange(self.leverage, self.leverage - 0.5, -0.02):
                    try:
                        size = round(self._amount * item, 0)
                        order = ex.create_market_order(self.symbol, side = "buy",
                                                     amount = size,
                                                     params = {"leverage" : self.leverage})
                        flag = False
                        break
                    except:
                        sleep(1)
                        pass
                if flag:
                    print("Couldn't Trade! Position = BUY")
                if flag == False:
                    self.print_status("buy", order)
                    self.units = abs(round(float(ex.fetch_positions(symbols = [self.symbol])[0]["info"]["positionAmt"]), 2))
                    self._real_units = self.units
                    self._notional = abs(float(ex.fetch_positions(symbols = [self.symbol])[0]["info"]["notional"]))
                    self._tp_units = 1
                    self.position = 1
            
        elif self._label == -1:
            if self.position == 1:
                order = ex.create_market_order(self.symbol, side = "sell", amount = self.units)
                self.print_status("close_b", order)
                self.position = 0
                sleep(1)
            if self.position == 0:
                flag = True
                for item in np.arange(self.leverage, self.leverage - 0.5, -0.02):
                    try:
                        size = round(self._amount * item, 0)
                        order = ex.create_market_order(self.symbol, side = "sell",
                                                     amount = size,
                                                     params = {"leverage" : self.leverage})
                        flag = False
                        break
                    except:
                        sleep(1)
                        pass
                if flag:
                    print("Couldn't Trade! Position = SELL")
                elif flag == False:
                    self.print_status("sell", order)
                    self.units = abs(round(float(ex.fetch_positions(symbols = [self.symbol])[0]["info"]["positionAmt"]), 2))
                    self._real_units = self.units
                    self._notional = abs(ex.fetch_positions(symbols = [self.symbol])[0]["notional"])
                    self._tp_units = 1
                    self.position = -1
        
    def tp_position(self):
        ex = self.binance
        if self.position in [-1, 1]:
            un_pnl = ex.fetch_positions(symbols = [self.symbol])[0]["unrealizedPnl"]
            notional = self._notional

            # ==========================================================================
            
            if ((un_pnl >= (notional * (5 / self.leverage) / 100)) and (self._tp_units == 1)):
                tp_units_5_percent = round((self._real_units * 4 / 10), 0)
                if self.position == 1:
                    order = ex.create_market_order(self.symbol, side = "sell",
                                                 amount = tp_units_5_percent)
                elif self.position == -1:
                    order = ex.create_market_order(self.symbol, side = "buy",
                                                 amount = tp_units_5_percent)
                self.units = self.units - tp_units_5_percent
                self._tp_units = 2
                self.tp_print_out(5, order)
            
            if ((un_pnl >= (notional * (7 / self.leverage) / 100)) and (self._tp_units == 2)):
                tp_units_7_percent = round((self._real_units * 3 / 10), 0)
                if self.position == 1:
                    order = ex.create_market_order(self.symbol, side = "sell",
                                                 amount = tp_units_7_percent)
                elif self.position == -1:
                    order = ex.create_market_order(self.symbol, side = "buy",
                                                 amount = tp_units_7_percent)
                self.units = self.units - tp_units_7_percent
                self._tp_units = 3
                self.tp_print_out(7, order)
            
            if ((un_pnl >= (notional * (10 / self.leverage) / 100)) and (self._tp_units == 3)):
                tp_units_10_percent = round((self._real_units * 2 / 10), 0)
                if self.position == 1:
                    order = ex.create_market_order(self.symbol, side = "sell",
                                                 amount = tp_units_10_percent)
                elif self.position == -1:
                    order = ex.create_market_order(self.symbol, side = "buy",
                                                 amount = tp_units_10_percent)
                self.units = self.units - tp_units_10_percent
                self._tp_units = 4
                self.tp_print_out(10, order)
                
    def c_number(self):
        self.n += 1
        print(f"{self.n}", end = " | ", flush = True)
        if self.position == 1:
            c_position = "LONG"
        elif self.position == -1:
            c_position = "SHORT"
        else:
            c_position = "NEUTRAL"
        if (self.n % 5) == 0:
            print("\n")
            print("=" * 50)
            print(f"{self.n}th Hour !", f"Current Position : {c_position}", sep = " ---> ")
            print("=" * 50)
            
    def print_status(self, status, order):
        ex = self.binance
        time = self.current_time()
        
        if "buy" in status:
            self.order_size_1 = round(float(ex.fetch_positions(symbols = [self.symbol])[0]["info"]["positionAmt"]), 2)
            print("-" * 100)
            print(f"{time} ---> Buying {self.symbol} for : ", self.order_size_1)
            print("-" * 100)
            
        elif "sell" in status:
            self.order_size_2 = round(float(ex.fetch_positions(symbols = [self.symbol])[0]["info"]["positionAmt"]), 2)
            print("-" * 100)
            print(f"{time} ---> Selling {self.symbol} for : ", self.order_size_2)
            print("-" * 100)
            
        elif "close_s" in status:
            order_size_3 = abs(float(ex.fetch_my_trades(symbol = self.symbol)[-1]["info"]["qty"]))
            order_size_3 = round(order_size_3, 2)
            pnl = abs(float(ex.fetch_my_trades(symbol = self.symbol)[-1]["info"]["realizedPnl"]))
            print("-" * 100)
            print(f"{time} ---> Closing SELL Position {self.symbol} | for : ", order_size_3)
            print(f"{time} ---> This Trade's PNL {pnl}")
            print(f"{time} ---> Cum P&L Until Now : {self.pnl}")
            print("-" * 100)
            
        elif "close_b" in status:
            order_size_3 = abs(float(ex.fetch_my_trades(symbol = self.symbol)[-1]["info"]["qty"]))
            order_size_3 = round(order_size_3, 2)
            pnl = abs(float(ex.fetch_my_trades(symbol = self.symbol)[-1]["info"]["realizedPnl"]))
            print("-" * 100)
            print(f"{time} ---> Closing BUY Position {self.symbol} | for : ", order_size_3)
            print(f"{time} ---> This Trade's PNL {pnl}")
            print(f"{time} ---> Cum P&L Until Now : {self.pnl}")
            print("-" * 100)
    
    def tp_print_out(self, going, order):
        ex = self.binance
        time = self.current_time()
        amount = float(order["info"]["origQty"])
        print("-" * 100)
        if going == 2:
            print(f"{time}  |  ORDER 2 PERCENT PLACED !")
            print(f"{time}  |  TAKE PROFIT ORDER AMOUNT IS : {amount}")
            print(f"{time}  |  THE REST AMOUNT IS : {self.units}")
        elif going == 5:
            print(f"{time}  |  ORDER 5 PERCENT PLACED !")
            print(f"{time}  |  TAKE PROFIT ORDER AMOUNT IS : {amount}")
            print(f"{time}  |  THE REST AMOUNT IS : {self.units}")
        elif going == 7:
            print(f"{time}  |  ORDER 7 PERCENT PLACED !")
            print(f"{time}  |  TAKE PROFIT ORDER AMOUNT IS : {amount}")
            print(f"{time}  |  THE REST AMOUNT IS : {self.units}")
        elif going == 10:
            print(f"{time}  |  ORDER 10 PERCENT PLACED !")
            print(f"{time}  |  TAKE PROFIT ORDER AMOUNT IS : {amount}")
            print(f"{time}  |  THE REST AMOUNT IS : {self.units}")
        print("-" * 100)
        
    def all_func(self):
        self.get_data()
        self.prepare_data()
        self.prepare_model()
        self.strategy()
        
    def refreshing_data(self):
        while True:
            tz_utc = pytz.timezone("UTC")
            current_time_utc = datetime.now(tz_utc)
            time = self.current_time()
            if (current_time_utc.minute % 15 == 0) and (current_time_utc.second > 5):
                print(f"{time}  ---->   Start Streaming !!!")
                print("=" * 110, end = "\n\n")
                break
            else:
                sleep(20)
        schedule.every(1).minutes.do(self.fetch_balances)
        schedule.every(15).minutes.do(self.all_func)
#         schedule.every(1).minutes.do(self.tp_position) # Enable it, if you want to have TP positions
        schedule.every(1).hours.do(self.c_number)
        while True:
            schedule.run_pending()
            sleep(10)
            
            
symbol = "LINK/USDT" # The currency pair you want to trade ---------> "BaseCurrency/QuoteCurrency"
api_key = "........." # The API key you got from Binance
secret_key = "..........." # The Secret key you got from Binance
testnet = False
model_path = "---" # The path of saved model in your local computer or server
scaler_path = "---" # The path of saved scaler in your local computer or server
bar_length = "15m" # timeframe
limit = 500 # Number of candles you need (based on how many NaN values you have)
leverage = 2 # Desired leverage


trader = BinanceTrader(symbol, api_key, secret_key, testnet, model_path, scaler_path, bar_length, limit, leverage)
print(trader)
thread = Thread(target = trader.refreshing_data)
thread.start()

