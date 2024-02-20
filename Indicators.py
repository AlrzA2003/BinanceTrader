import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class Indicators:
    def __init__(self, df):
        self._df = df
        self.data = df
    def __repr__(self):
        return "Class : Indicators  |  Contains several indicators for Neural Network Processing!"
    
    # First 10 Indicators !!!  ---------------------------------------------
    
    def macd(self, data, short_period, long_period, signal_period):
        ema_short = data['Close'].ewm(span=short_period, adjust=False).mean()
        ema_long = data['Close'].ewm(span=long_period, adjust=False).mean()
        macd = ema_short - ema_long
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        histogram = macd - signal
        return (macd, signal, histogram)
    
    def rsi(self, data, period): # 14
        delta = data['Close'].diff()
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        gain = up.rolling(window=period).mean()
        loss = abs(down.rolling(window=period).mean())
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def stochastic(self, data, period): # 14
        low_min = data['Low'].rolling(window=period).min()
        high_max = data['High'].rolling(window=period).max()
        k = 100 * ((data['Close'] - low_min) / (high_max - low_min))
        d = k.rolling(window=3).mean()
        return (k, d)

    def roc(self, data, period): # 12
        roc = 100 * (data['Close'] - data['Close'].shift(period)) / data['Close'].shift(period)
        return roc
    
    def atr(self, data, period): # 14
        high_low = data['High'] - data['Low']
        high_close = abs(data['High'] - data['Close'].shift())
        low_close = abs(data['Low'] - data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def SuperTrend(self, data, atr_period, multiplier): # 5, 1
        atr = self.atr(data, atr_period)
        upper = ((data["High"] + data["Low"]) / 2) + (multiplier * atr)
        lower = ((data["High"] + data["Low"]) / 2) - (multiplier * atr)
        final_uppertrend = list(upper.values)
        final_lowertrend = list(lower.values)
        sp_trend = [True] * len(data)
        for bar in range(1, len(data)):
            curr, prev = bar, bar - 1
            if data.iloc[curr, list(data.columns).index("Close")] > final_uppertrend[prev]:
                sp_trend[curr] = True
            elif data.iloc[curr, list(data.columns).index("Close")] < final_lowertrend[prev]:
                sp_trend[curr] = False
            else:
                sp_trend[curr] = sp_trend[prev]
                if sp_trend[curr] == False:
                    c_up = final_uppertrend[curr]
                    p_up = final_uppertrend[prev]
                    if c_up > p_up:
                        final_uppertrend[curr] = p_up
                if (sp_trend[curr] == True) and (final_lowertrend[curr] < final_lowertrend[prev]):
                    final_lowertrend[curr] = final_lowertrend[prev]
            if sp_trend[curr] == True:
                final_uppertrend[curr] = np.nan
            elif sp_trend[curr] == False:
                final_lowertrend[curr] = np.nan
        upper = final_uppertrend
        lower = final_lowertrend
        sp_trend = sp_trend
        return sp_trend
    
    
    def trix(self, data, period):
        """
        Calculates TRIX given closing prices and period.
        """
        ema1 = data["Close"].ewm(span=period, min_periods=period).mean()
        ema2 = ema1.ewm(span=period, min_periods=period).mean()
        ema3 = ema2.ewm(span=period, min_periods=period).mean()
        trix = ((ema3 - ema3.shift(1)) / ema3.shift(1)) * 100

        return trix
        
    def ElderRay(self, data, period=13):
        data['EMA'] = data['Close'].ewm(span=period).mean()
        bull_power = pd.Series(data['High'] - data['EMA'], name='Bull Power')
        bear_power = pd.Series(data['Low'] - data['EMA'], name='Bear Power')
        return (bull_power, bear_power)
        
    def true_strength_index(self, df, r=25, s=13):
        m = df['Close'].diff()
        m_abs = m.abs()
        ema1 = m.ewm(span=r).mean()
        ema2 = ema1.ewm(span=s).mean()
        ema3 = m_abs.ewm(span=r).mean()
        ema4 = ema3.ewm(span=s).mean()
        tsi = 100 * (ema2 / ema4)
        return tsi
    
    def chande_momentum_oscillator(self, df, period=20):
        close = df['Close']
        diff = close.diff()
        su = diff.where(diff > 0, 0.0).rolling(window = period).sum()
        sd = -diff.where(diff < 0, 0.0).rolling(window = period).sum()
        cmo = 100 * (su - sd) / (su + sd)
        return cmo
    
    
    # ===============================================================================================
    # ===============================================================================================
    # ===============================================================================================
    def all_ind(self):
        data = self.data.copy()
        # MACD ******************************************************************************************************************
        MACD = self.macd(data = data, short_period = 12, long_period = 26, signal_period = 9)
        data["MACD_1"], data["signal_1"], data["histogram_1"] = MACD[0], MACD[1], MACD[2]
        # ===========
        MACD = self.macd(data = data, short_period = 9, long_period = 21, signal_period = 10)
        data["MACD_2"], data["signal_2"], data["histogram_2"] = MACD[0], MACD[1], MACD[2]
        # ==========
        MACD = self.macd(data = data, short_period = 15, long_period = 30, signal_period = 10)
        data["MACD_3"], data["signal_3"], data["histogram_3"] = MACD[0], MACD[1], MACD[2]
        # ==========
        MACD = self.macd(data = data, short_period = 12, long_period = 20, signal_period = 9)
        data["MACD_4"], data["signal_4"], data["histogram_4"] = MACD[0], MACD[1], MACD[2]
        # ==========
        MACD = self.macd(data = data, short_period = 15, long_period = 26, signal_period = 12)
        data["MACD_5"], data["signal_5"], data["histogram_5"] = MACD[0], MACD[1], MACD[2]
        
        # RSI *******************************************************************************************************************
        
        data["RSI"] = self.rsi(data = data, period = 8)
                
        # STOCHASTIC ************************************************************************************************************

        STOCHASTIC = self.stochastic(data = data, period = 4)
        data["stochastic_K_2"], data["stochastic_D_2"] = STOCHASTIC[0], STOCHASTIC[1]
        
        # ROC *******************************************************************************************************************

        data["ROC_1"] = self.roc(data = data, period = 8)
        # =========
        data["ROC_3"] = self.roc(data = data, period = 18)
        
        # SP_Trend **************************************************************************************************************
        
        sp_trend = pd.get_dummies(pd.Series(self.SuperTrend(data = data, atr_period = 7, multiplier = 1), index = data.index))[True]
        data["SP_trend_1"] = sp_trend
        # =========
        sp_trend = pd.get_dummies(pd.Series(self.SuperTrend(data = data, atr_period = 10, multiplier = 1), index = data.index))[True]
        data["SP_trend_2"] = sp_trend
        # =========
        sp_trend = pd.get_dummies(pd.Series(self.SuperTrend(data = data, atr_period = 10, multiplier = 2), index = data.index))[True]
        data["SP_trend_3"] = sp_trend
        
        # TRIX ******************************************************************************************************************
        
        data["TRIX_1"] = self.trix(data = data, period = 15)
        # =========
        data["TRIX_2"] = self.trix(data = data, period = 12)
        # =========
        data["TRIX_3"] = self.trix(data = data, period = 21)
                                
        # Elder_Ray *********************************************************************************************************
        
        data["Elder_Bull_1"], data["Elder_Bear_1"] = self.ElderRay(data = data, period = 13)
        # =========
        data["Elder_Bull_2"], data["Elder_Bear_2"] = self.ElderRay(data = data, period = 24)
        # =========
        data["Elder_Bull_3"], data["Elder_Bear_3"] = self.ElderRay(data = data, period = 20)
        
        # -------------------------------------------------------
                
        # TSI **************************************************************************************************************
        
        data["true_strength_index_1"] = self.true_strength_index(df = data, r = 25, s = 13)
        # =========
        data["true_strength_index_2"] = self.true_strength_index(df = data, r = 15, s = 22)
        # =========
        data["true_strength_index_3"] = self.true_strength_index(df = data, r = 30, s = 18)
        
        # chande_momentum **************************************************************************************************
        
        data["chande_momentum_1"] = self.chande_momentum_oscillator(df = data, period = 20)
        # =========
        data["chande_momentum_2"] = self.chande_momentum_oscillator(df = data, period = 32)
        # =========
        data["chande_momentum_3"] = self.chande_momentum_oscillator(df = data, period = 27)        

        self.data = data
        
    # ANOTHER TYPE OF RETURN --------------------------------
        
    def add_pca(self, num_of_bunches = 1):
        data = self.data.copy()
        scaler = StandardScaler()
        try:
            pre_x = data.drop(["Open", "High", "Low", "Close", "Volume"], axis = 1).dropna()
            x = pre_x.values
            inds = pre_x.index
        except:
            pre_x = data.drop(["Open", "High", "Low", "Close"], axis = 1).dropna()
            x = pre_x.values
            inds = pre_x.index
        scaled_data = scaler.fit_transform(x)
        pca = PCA(num_of_bunches)
        x_pca = pca.fit_transform(scaled_data)
        x_pca = x_pca.reshape(len(x_pca))
        na = np.array((len(data) - len(x_pca)) * [np.nan])
        x_pca = np.concatenate((na, x_pca))
        s = pd.Series(x_pca, index = data.index)
        return s
