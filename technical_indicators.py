import numpy as np
import pandas as pd


def calculate_technical_indicators(price_data):

    price_data = calculate_williams_percent_r(price_data,14)
    price_data = calculate_rate_of_change(price_data,14)
    price_data = calculate_rsi(price_data,7)
    price_data = calculate_rsi(price_data,14)
    price_data = calculate_rsi(price_data,28)
    price_data = calculate_macd(price_data, 8, 21)
    price_data = calculate_bollinger_bands(price_data,20)    
    price_data = calculate_ichimoku_cloud(price_data)
    price_data = calculate_exponential_moving_average(price_data, 3)
    price_data = calculate_exponential_moving_average(price_data, 8)
    price_data = calculate_exponential_moving_average(price_data, 15)
    price_data = calculate_exponential_moving_average(price_data, 50)
    price_data = calculate_exponential_moving_average(price_data, 100)
    price_data = calculate_average_directional_index(price_data, 14)
    price_data = calculate_donchian_channels(price_data, 10)
    price_data = calculate_donchian_channels(price_data, 20)
    price_data = calculate_arnaud_legoux_moving_average(price_data, 10)
    price_data = calculate_true_strength_index(price_data, 13, 25)
    price_data = calculate_zscore(price_data, 20)
    price_data = calculate_log_return(price_data, 10)
    price_data = calculate_log_return(price_data, 20)
    price_data = calculate_vortex_indicator(price_data, 7)
    price_data = calculate_aroon_indicator(price_data, 16)
    price_data = calculate_elder_bull_bear_power(price_data, 14)
    price_data = calculate_acceleration_bands(price_data, 20)
    price_data = calculate_short_run(price_data, 14)
    price_data = calculate_bias(price_data, 26)
    price_data = calculate_ttm_trend(price_data, 5, 20)
    price_data = calculate_percent_return(price_data, 10)
    price_data = calculate_percent_return(price_data, 20)
    price_data = calculate_kurtosis(price_data, 5)
    price_data = calculate_kurtosis(price_data, 10)
    price_data = calculate_kurtosis(price_data, 20)
    price_data = calculate_elder_force_index(price_data, 13)    
    price_data = calculate_average_true_range(price_data, 14)
    price_data = calculate_keltner_channels(price_data, 20)
    price_data = calculate_chaikin_volatility(price_data, 10)
    price_data = calculate_standard_deviation(price_data, 5)
    price_data = calculate_standard_deviation(price_data, 10)
    price_data = calculate_standard_deviation(price_data, 20)
    price_data = calculate_volatility_index(price_data, 21)    
    price_data = calculate_on_balance_volume(price_data, 10)
    price_data = calculate_chaikin_money_flow(price_data, 5)
    price_data = calculate_volume_price_trend(price_data, 7)
    price_data = calculate_accumulation_distribution_line(price_data, 3)
    price_data = calculate_ease_of_movement(price_data, 14)
    
    return price_data

# Williams %R
def calculate_williams_percent_r(price_data, window=14):
    highest_high = price_data["High"].rolling(window=window).max()
    lowest_low = price_data["Low"].rolling(window=window).min()
    price_data["Williams_%R{}".format(window)] = -((highest_high - price_data["Close"]) / (highest_high - lowest_low)) * 100
    return price_data

# Rate of Change
def calculate_rate_of_change(price_data, window=14):
    price_data["ROC_{}".format(window)] = (price_data["Close"] / price_data["Close"].shift(window) - 1) * 100
    return price_data

# RSI
def calculate_rsi(price_data, window=14) : 
    delta = price_data["Close"].diff(1)
    gains = delta.where(delta>0,0)
    losses = -delta.where(delta<0,0)
    avg_gain = gains.rolling(window=window, min_periods=1).mean()
    avg_loss = losses.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    price_data["rsi_{}".format(window)] = 100 - (100 / (1 + rs))
    return price_data

# MACD 
def calculate_macd(price_data, short_window=8, long_window=21, signal_window=9):
    short_ema = price_data["Close"].ewm(span = short_window, adjust = False).mean()
    long_ema = price_data["Close"].ewm(span = long_window, adjust = False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    price_data["MACD_Line"] = macd_line
    price_data["Signal_Line"] = signal_line
    price_data["MACD_Histogram"] = macd_histogram
    return price_data

# Bollinger Bands
def calculate_bollinger_bands(price_data, window=20, num_std_dev=2) :
    price_data["midlle_band"] = price_data["Close"].rolling(window=window).mean()
    price_data["std"] = price_data["Close"].rolling(window=window).std()
    price_data["upper_band{}".format(window)] = price_data["midlle_band"] + (num_std_dev * price_data["std"])
    price_data["lower_band{}".format(window)] = price_data["midlle_band"] - (num_std_dev * price_data["std"])
    price_data.drop(["std"], axis=1, inplace=True)   
    return price_data

# Ichimoku Cloud
def calculate_ichimoku_cloud(price_data, window_tenkan=9, window_kijun=26, window_senkou_span_b=52, window_chikou=26):
    tenkan_sen = (price_data["Close"].rolling(window=window_tenkan).max() + price_data["Close"].rolling(window=window_tenkan).min()) / 2
    kijun_sen = (price_data["Close"].rolling(window=window_kijun).max() + price_data["Close"].rolling(window=window_kijun).min()) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(window_kijun)
    senkou_span_b = (price_data["Close"].rolling(window=window_senkou_span_b).max() + price_data["Close"].rolling(window=window_senkou_span_b).min()) / 2
    chikou_span = price_data["Close"].shift(-window_chikou)
    price_data["Tenkan_sen"] = tenkan_sen
    price_data["Kijun_sen"] = kijun_sen
    price_data["Senkou_Span_A"] = senkou_span_a
    price_data["Senkou_Span_B"] = senkou_span_b
    price_data["Chikou_Span"] = chikou_span
    return price_data

# Moving Average (EMA)
def calculate_exponential_moving_average(price_data, window=8): 
    price_data["ema_{}".format(window)] = price_data["Close"].ewm(span=window, adjust=False).mean()
    return price_data

# ADX
def calculate_average_directional_index(price_data, window=14): #14
    price_data["TR"] = abs(price_data["High"] - price_data["Low"]).combine_first(abs(price_data["High"] - price_data["Close"].shift(1))).combine_first(abs(price_data["Low"] - price_data["Close"].shift(1)))
    price_data["DMplus"] = (price_data["High"] - price_data["High"].shift(1)).apply(lambda x: x if x > 0 else 0)
    price_data["DMminus"] = (price_data["Low"].shift(1) - price_data["Low"]).apply(lambda x: x if x > 0 else 0)
    price_data["ATR"] = price_data["TR"].rolling(window=window).mean()
    price_data["DIplus"] = (price_data["DMplus"].rolling(window=window).mean() / price_data["ATR"]) * 100
    price_data["DIminus"] = (price_data["DMminus"].rolling(window=window).mean() / price_data["ATR"]) * 100
    price_data["DX"] = abs(price_data["DIplus"] - price_data["DIminus"]) / (price_data["DIplus"] + price_data["DIminus"]) * 100
    price_data["ADX_{}".format(window)] = price_data["DX"].rolling(window=window).mean()
    price_data.drop(["TR", "DMplus", "DMminus", "ATR", "DIplus", "DIminus", "DX"], axis=1, inplace=True)
    return price_data

# Donchian Channel
def calculate_donchian_channels(price_data, window=10):
    highest_high = price_data["Close"].rolling(window=window).max()
    lowest_low = price_data["Close"].rolling(window=window).min()
    price_data["Donchian_Upper_{}".format(window)] = highest_high
    price_data["Donchian_Lower_{}".format(window)] = lowest_low
    return price_data

# Arnaud Legoux Moving Average (ALMA)
def calculate_arnaud_legoux_moving_average(price_data, window=10, sigma=6, offset=0.85):
    m = np.linspace(-offset*(window-1), offset*(window-1), window)
    w = np.exp(-0.5 * (m / sigma) ** 2)
    w /= w.sum()
    alma_values = np.convolve(price_data["Close"].values, w, mode="valid")
    alma_values = np.concatenate([np.full(window-1, np.nan), alma_values])
    price_data["ALMA_{}".format(window)] = alma_values
    return price_data

# True Strength Index (TSI)
def calculate_true_strength_index(price_data, short_period=13, long_period=25):
    price_diff = price_data["Close"].diff(1)
    double_smoothed = price_diff.ewm(span=short_period, min_periods=1, adjust=False).mean().ewm(span=long_period, min_periods=1, adjust=False).mean()
    double_smoothed_abs = price_diff.abs().ewm(span=short_period, min_periods=1, adjust=False).mean().ewm(span=long_period, min_periods=1, adjust=False).mean()
    tsi_values = 100 * double_smoothed / double_smoothed_abs
    price_data["TSI_{}_{}".format(short_period, long_period)] = tsi_values
    return price_data

# Z-Score
def calculate_zscore(price_data, window=20):
    rolling_mean = price_data["Close"].rolling(window=window).mean()
    rolling_std = price_data["Close"].rolling(window=window).std()
    z_score = (price_data["Close"] - rolling_mean) / rolling_std
    price_data["Z_Score_{}".format(window)] = z_score
    return price_data

# Log Return
def calculate_log_return(price_data, window=5):
    price_data["LogReturn_{}".format(window)] = price_data["Close"].pct_change(window).apply(lambda x: 0 if pd.isna(x) else x)
    return price_data

# Vortex Indicator
def calculate_vortex_indicator(price_data, window=7): 
    high_low = price_data["High"] - price_data["Low"]
    high_close_previous = abs(price_data["High"] - price_data["Close"].shift(1))
    low_close_previous = abs(price_data["Low"] - price_data["Close"].shift(1))
    true_range = pd.concat([high_low, high_close_previous, low_close_previous], axis=1).max(axis=1)
    positive_vm = abs(price_data["High"].shift(1) - price_data["Low"])
    negative_vm = abs(price_data["Low"].shift(1) - price_data["High"])
    true_range_sum = true_range.rolling(window=window).sum()
    positive_vm_sum = positive_vm.rolling(window=window).sum()
    negative_vm_sum = negative_vm.rolling(window=window).sum()
    positive_vi = positive_vm_sum / true_range_sum
    negative_vi = negative_vm_sum / true_range_sum
    price_data["Positive_VI_{}".format(window)] = positive_vi
    price_data["Negative_VI_{}".format(window)] = negative_vi
    return price_data

# Aroon Indicator
def calculate_aroon_indicator(price_data, window=16):
    high_prices = price_data["High"]
    low_prices = price_data["Low"]
    aroon_up = []
    aroon_down = []
    for i in range(window, len(high_prices)):
        high_period = high_prices[i - window:i + 1]
        low_period = low_prices[i - window:i + 1]
        high_index = window - high_period.values.argmax() - 1
        low_index = window - low_period.values.argmin() - 1
        aroon_up.append((window - high_index) / window * 100)
        aroon_down.append((window - low_index) / window * 100)
    aroon_up = [None] * window + aroon_up
    aroon_down = [None] * window + aroon_down
    price_data["Aroon_Up_{}".format(window)] = aroon_up
    price_data["Aroon_Down_{}".format(window)] = aroon_down
    return price_data

# Elder"s Bull Power e Bear Power 
def calculate_elder_bull_bear_power(price_data, window=14):
    ema = price_data["Close"].ewm(span=window, adjust=False).mean()
    bull_power = price_data["High"] - ema
    bear_power = price_data["Low"] - ema
    price_data["Bull_Power_{}".format(window)] = bull_power
    price_data["Bear_Power_{}".format(window)] = bear_power
    return price_data

# Acceleration Bands
def calculate_acceleration_bands(price_data, window=20, acceleration_factor=0.02):
    sma = price_data["Close"].rolling(window=window).mean()
    band_difference = price_data["Close"] * acceleration_factor
    upper_band = sma + band_difference
    lower_band = sma - band_difference
    price_data["Upper_Band_{}".format(window)] = upper_band
    price_data["Lower_Band_{}".format(window)] = lower_band
    price_data["Middle_Band_{}".format(window)] = sma
    return price_data

# Short Run
def calculate_short_run(price_data, window=14):
    short_run = price_data["Close"] - price_data["Close"].rolling(window=window).min()
    price_data["Short_Run_{}".format(window)] = short_run
    return price_data

# Bias
def calculate_bias(price_data, window=26):
    moving_average = price_data["Close"].rolling(window=window).mean()
    bias = ((price_data["Close"] - moving_average) / moving_average) * 100
    price_data["Bias_{}".format(window)] = bias
    return price_data

# TTM Trend
def calculate_ttm_trend(price_data, short_window=5, long_window=20):
    short_ema = price_data["Close"].ewm(span=short_window, adjust=False).mean()
    long_ema = price_data["Close"].ewm(span=long_window, adjust=False).mean()
    ttm_trend = short_ema - long_ema
    price_data["TTM_Trend_{}_{}".format(short_window, long_window)] = ttm_trend
    return price_data

# Percent Return
def calculate_percent_return(price_data, window=1): 
    percent_return = price_data["Close"].pct_change().rolling(window=window).mean() * 100
    price_data["Percent_Return_{}".format(window)] = percent_return
    return price_data

# Kurtosis
def calculate_kurtosis(price_data, window=20):
    price_data["kurtosis_{}".format(window)] = price_data["Close"].rolling(window=window).apply(lambda x: np.nan if x.isnull().any() else x.kurt())
    return price_data

# Elder's Force Index (ERI)
def calculate_elder_force_index(price_data, window=13):
    price_change = price_data["Close"].diff()
    force_index = price_change * price_data["Volume"]
    eri = force_index.ewm(span=window, adjust=False).mean()
    price_data["ERI_{}".format(window)] = eri
    return price_data

# ATR
def calculate_average_true_range(price_data, window=14):
    price_data["High-Low"] = price_data["High"] - price_data["Low"]
    price_data["High-PrevClose"] = abs(price_data["High"] - price_data["Close"].shift(1))
    price_data["Low-PrevClose"] = abs(price_data["Low"] - price_data["Close"].shift(1))
    price_data["TrueRange"] = price_data[["High-Low", "High-PrevClose", "Low-PrevClose"]].max(axis=1)
    price_data["atr_{}".format(window)] = price_data["TrueRange"].rolling(window=window, min_periods=1).mean()
    price_data.drop(["High-Low", "High-PrevClose", "Low-PrevClose", "TrueRange"], axis=1, inplace=True)
    return price_data

# Keltner Channels
def calculate_keltner_channels(price_data, period=20, multiplier=2):
    price_data["TR"] = price_data.apply(lambda row: max(row["High"] - row["Low"], abs(row["High"] - row["Close"]), abs(row["Low"] - row["Close"])), axis=1)
    price_data["ATR"] = price_data["TR"].rolling(window=period).mean()
    price_data["Middle Band"] = price_data["Close"].rolling(window=period).mean()
    price_data["Upper Band"] = price_data["Middle Band"] + multiplier * price_data["ATR"]
    price_data["Lower Band"] = price_data["Middle Band"] - multiplier * price_data["ATR"]
    return price_data

# Chaikin Volatility
def calculate_chaikin_volatility(price_data, window=10):
    daily_returns = price_data["Close"].pct_change()
    chaikin_volatility = daily_returns.rolling(window=window).std() * (252 ** 0.5)
    price_data["Chaikin_Volatility_{}".format(window)] = chaikin_volatility
    return price_data

# Standard Deviation 
def calculate_standard_deviation(price_data, window=1): 
    stdev_column = price_data["Close"].rolling(window=window).std()
    price_data["Stdev_{}".format(window)] = stdev_column
    return price_data

# Volatility Index (VIX)
def calculate_volatility_index(price_data, window=21):
    returns = price_data["Close"].pct_change().dropna()
    rolling_std = returns.rolling(window=window).std()
    vix = rolling_std * np.sqrt(252) * 100  
    price_data["VIX_{}".format(window)] = vix
    return price_data

# On-Balance Volume (OBV)
def calculate_on_balance_volume(price_data, window=10):
    price_changes = price_data["Close"].diff()
    volume_direction = pd.Series(1, index=price_changes.index)
    volume_direction[price_changes < 0] = -1
    obv = (price_data["Volume"] * volume_direction).cumsum()
    obv_smoothed = obv.rolling(window=window).mean()
    price_data["OBV_{}".format(window)] = obv_smoothed
    return price_data

# Chaikin Money Flow (CMF)
def calculate_chaikin_money_flow(price_data, window=10):
    mf_multiplier = ((price_data["Close"] - price_data["Close"].shift(1)) + (price_data["Close"] - price_data["Close"].shift(1)).abs()) / 2
    mf_volume = mf_multiplier * price_data["Volume"]
    adl = mf_volume.cumsum()
    cmf = adl.rolling(window=window).mean() / price_data["Volume"].rolling(window=window).mean()
    price_data["CMF_{}".format(window)] = cmf
    return price_data

# Volume Price Trend (VPT)
def calculate_volume_price_trend(price_data, window=10):
    price_change = price_data["Close"].pct_change()
    vpt = (price_change * price_data["Volume"].shift(window)).cumsum()
    price_data["VPT_{}".format(window)] = vpt
    return price_data

# Accumulation/Distribution Line
def calculate_accumulation_distribution_line(price_data, window=10):
    money_flow_multiplier = ((price_data["Close"] - price_data["Close"].shift(1)) - (price_data["Close"].shift(1) - price_data["Close"])) / (price_data["Close"].shift(1) - price_data["Close"])
    money_flow_volume = money_flow_multiplier * price_data["Volume"]
    ad_line = money_flow_volume.cumsum()
    ad_line_smoothed = ad_line.rolling(window=window, min_periods=1).mean()
    price_data["A/D Line_{}".format(window)] = ad_line_smoothed
    return price_data

# Ease of Movement (EOM)
def calculate_ease_of_movement(price_data, window=14):
    midpoint_move = ((price_data["High"] + price_data["Low"]) / 2).diff(1)
    box_ratio = price_data["Volume"] / 1000000 / (price_data["High"] - price_data["Low"])
    eom = midpoint_move / box_ratio
    eom_smoothed = eom.rolling(window=window, min_periods=1).mean()
    price_data["EOM_{}".format(window)] = eom_smoothed
    return price_data
    

