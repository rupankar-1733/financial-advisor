import pandas as pd
import numpy as np

def calculate_rsi(data, window=14):
    """
    Calculate Relative Strength Index (RSI)
    
    RSI measures overbought/oversold conditions
    - RSI > 70: Potentially overbought (sell signal)
    - RSI < 30: Potentially oversold (buy signal)
    
    Parameters:
    data: pandas Series of close prices
    window: int, period for RSI calculation (default 14)
    
    Returns:
    pandas Series with RSI values (0-100)
    """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_stochastic(high, low, close, k_window=14, d_window=3):
    """
    Calculate Stochastic Oscillator (%K and %D)
    
    Compares closing price to price range over given period
    - %K > 80: Overbought
    - %K < 20: Oversold
    
    Parameters:
    high, low, close: pandas Series of OHLC data
    k_window: int, period for %K calculation
    d_window: int, period for %D (SMA of %K)
    
    Returns:
    dict with 'K' and 'D' pandas Series
    """
    lowest_low = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()
    
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_window).mean()
    
    return {'K': k_percent, 'D': d_percent}

def calculate_bollinger_bands(data, window=20, num_std=2):
    """
    Calculate Bollinger Bands
    
    Middle Band: 20-day SMA
    Upper Band: SMA + (2 × Standard Deviation)
    Lower Band: SMA - (2 × Standard Deviation)
    
    Trading Signals:
    - Price touches upper band: Potential sell
    - Price touches lower band: Potential buy
    - Squeeze (bands narrow): Low volatility, breakout expected
    
    Parameters:
    data: pandas Series of close prices
    window: int, period for moving average (default 20)
    num_std: float, number of standard deviations (default 2)
    
    Returns:
    dict with 'upper', 'middle', 'lower' pandas Series
    """
    sma = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    
    return {
        'upper': upper_band,
        'middle': sma,
        'lower': lower_band
    }

def generate_rsi_signals(rsi, overbought=70, oversold=30):
    """
    Generate trading signals based on RSI levels
    
    Parameters:
    rsi: pandas Series of RSI values
    overbought: float, RSI level for overbought (default 70)
    oversold: float, RSI level for oversold (default 30)
    
    Returns:
    pandas Series with signals: 1 for buy, -1 for sell, 0 for hold
    """
    signals = pd.Series(0, index=rsi.index)
    
    # Buy signal: RSI crosses above oversold level
    buy_signals = (rsi > oversold) & (rsi.shift(1) <= oversold)
    
    # Sell signal: RSI crosses below overbought level
    sell_signals = (rsi < overbought) & (rsi.shift(1) >= overbought)
    
    signals[buy_signals] = 1
    signals[sell_signals] = -1
    
    return signals

def calculate_momentum(data, window=10):
    """
    Calculate Price Momentum
    
    Momentum = Current Price - Price N periods ago
    
    Parameters:
    data: pandas Series of close prices
    window: int, number of periods to look back
    
    Returns:
    pandas Series with momentum values
    """
    return data - data.shift(window)
