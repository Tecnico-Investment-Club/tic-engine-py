import pandas as pd
import numpy as np


import pandas as pd

def compute_macd_initial(price, fast=12, slow=26, signal=9):
    """
    Computes MACD line and Signal line.
    - price: pd.Series of prices
    - fast: short EMA window (default 12)
    - slow: long EMA window (default 26)
    - signal: signal line EMA window (default 9)
    Returns: pd.DataFrame with columns ['macd', 'signal', 'hist']
    """
    ema_fast = price.ewm(span=fast, adjust=False).mean()
    ema_slow = price.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - signal_line
    macd_df = pd.DataFrame({
        'macd': macd,
        'signal': signal_line,
        'hist': macd_hist}, index=price.index)
    return macd_df, ema_fast, ema_slow, signal_line


def compute_rsi_initial(price, window=14):
    """
    Computes the Relative Strength Index (RSI).
    - price: pd.Series of prices
    - window: lookback period (default 14)
    Returns: pd.DataFrame with column ['rsi'] and other auxiliary values.
    """
    delta = price.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    avg_gain = up.ewm(alpha=1/window, adjust=False).mean()
    avg_loss = down.ewm(alpha=1/window, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-10)  # add small constant to avoid div by zero
    rsi = 100 - (100 / (1 + rs))
    
    rsi_df = pd.DataFrame({'rsi': rsi}, index=price.index)
    return rsi_df, avg_gain, avg_loss, price.iloc[-1] 


def rolling_return_initial(series, window):
    """
    Computes the rolling percentage change (returns).
    - series: pd.Series of values
    - window: lookback period for the percentage change
    Returns: pd.DataFrame with column ['rolling_return'] and the raw series.
    """
    rolling_ret = series.pct_change(periods=window)
    rolling_return_df = pd.DataFrame({'rolling_return': rolling_ret}, index=series.index)
    return rolling_return_df, series


def z_score_initial(series, window):
    """
    Computes the Z-score (standardized value) for a rolling window.
    - series: pd.Series of values
    - window: lookback period for mean and standard deviation calculation
    Returns: pd.DataFrame with column ['z_score'] and the mean and standard deviation.
    """
    mean = series.rolling(window=window).mean()
    std = series.rolling(window=window).std(ddof=0)
    z_score_val = (series - mean) / std
    z_score_df = pd.DataFrame({'z_score': z_score_val}, index=series.index)
    return z_score_df, mean, std

def update_macd(new_prices, last_ema_fast, last_ema_slow, last_signal, fast=12, slow=26, signal=9):
    """
    Incrementally computes MACD values for a date-indexed price series.
    Assumes previous EMA and signal values are provided (no internal init).

    Args:
        new_prices (pd.Series): Price series with datetime index.
        last_ema_fast (float): Previous fast EMA.
        last_ema_slow (float): Previous slow EMA.
        last_signal (float): Previous signal EMA.
        fast (int): Fast EMA span (default 12).
        slow (int): Slow EMA span (default 26).
        signal (int): Signal line EMA span (default 9).

    Returns:
        df (pd.DataFrame): DataFrame with datetime index and columns ['macd', 'signal', 'hist'].
        last_ema_fast, last_ema_slow, last_signal: Updated states.
    """

    alpha_fast = 2 / (fast + 1)
    alpha_slow = 2 / (slow + 1)
    alpha_signal = 2 / (signal + 1)

    results = []

    for date, price in new_prices.items():
        last_ema_fast = alpha_fast * price + (1 - alpha_fast) * last_ema_fast
        last_ema_slow = alpha_slow * price + (1 - alpha_slow) * last_ema_slow
        macd = last_ema_fast - last_ema_slow
        last_signal = alpha_signal * macd + (1 - alpha_signal) * last_signal
        hist = macd - last_signal
        results.append((date, macd, last_signal, hist))

    df = pd.DataFrame(results, columns=['date', 'macd', 'signal', 'hist']).set_index('date')
    return df, last_ema_fast, last_ema_slow, last_signal

def update_rsi(new_prices, prev_avg_gain, prev_avg_loss, prev_price, window=14):
    """
    Incrementally computes RSI values for new prices using previous EMA states.

    Args:
        new_prices (pd.Series): New prices with datetime index.
        prev_avg_gain (float): Previous average gain (None if not initialized).
        prev_avg_loss (float): Previous average loss (None if not initialized).
        prev_price (float): Previous price to compute the first delta.
        window (int): RSI window size (default 14).

    Returns:
        df (pd.DataFrame): DataFrame with datetime index and 'rsi' column.
        new_avg_gain, new_avg_loss, last_price: Updated states.
    """

    alpha = 1 / window
    results = []

    for date, price in new_prices.items():
        delta = price - prev_price
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        prev_avg_gain = alpha * gain + (1 - alpha) * prev_avg_gain
        prev_avg_loss = alpha * loss + (1 - alpha) * prev_avg_loss

        rs = prev_avg_gain / (prev_avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        results.append((date, rsi))

        prev_price = price

    df = pd.DataFrame(results, columns=["date", "rsi"]).set_index("date")
    return df, prev_avg_gain, prev_avg_loss, prev_price


def update_rolling_return(new_prices: pd.Series, prev_prices: pd.Series, window: int):
    """
    Incrementally computes rolling returns from previous prices and new data.

    Args:
        new_prices (pd.Series): New prices with datetime index.
        prev_prices (pd.Series): Previous tail (at most `window` old) with datetime index.
        window (int): Rolling window size.

    Returns:
        result_df (pd.DataFrame): DataFrame with 'rolling_return' and datetime index.
        updated_tail (pd.Series): New tail to use in next call.
    """
    combined = pd.concat([prev_prices, new_prices])
    rolling_ret = combined.pct_change(periods=window)

    result = rolling_ret.loc[new_prices.index]
    updated_tail = combined.iloc[-window:]
    return pd.DataFrame({'rolling_return': result}), updated_tail


def update_z_score(new_prices: pd.Series, prev_prices: pd.Series, window: int):
    """
    Incrementally computes z-score from previous prices and new data.

    Args:
        new_prices (pd.Series): New prices with datetime index.
        prev_prices (pd.Series): Previous tail (at most `window-1` old) with datetime index.
        window (int): Rolling window size.

    Returns:
        result_df (pd.DataFrame): DataFrame with 'z_score' and datetime index.
        updated_tail (pd.Series): New tail to use in next call.
    """
    combined = pd.concat([prev_prices, new_prices])
    mean = combined.rolling(window=window).mean()
    std = combined.rolling(window=window).std(ddof=0)
    z_score = (combined - mean) / (std + 1e-10)

    result = z_score.loc[new_prices.index]
    updated_tail = combined.iloc[-(window - 1):]
    return pd.DataFrame({'z_score': result}), updated_tail

