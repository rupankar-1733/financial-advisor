import pandas as pd
import numpy as np

def calculate_sma(data, window):
    """
    Calculate Simple Moving Average (SMA)
    
    Parameters:
    data: pandas Series of prices (usually Close prices)  
    window: int, number of periods for moving average
    
    Returns:
    pandas Series with SMA values
    
    Example: SMA(20) = average of last 20 closing prices
    """
    return data.rolling(window=window).mean()

def calculate_ema(data, window):
    """
    Calculate Exponential Moving Average (EMA)
    More weight to recent prices - responsive to recent changes
    
    Parameters:
    data: pandas Series of prices
    window: int, number of periods for EMA
    
    Returns:
    pandas Series with EMA values
    
    Formula: EMA = (Close × Multiplier) + (Previous_EMA × (1 - Multiplier))
    Multiplier = 2 / (window + 1)
    """
    return data.ewm(span=window, adjust=False).mean()

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate MACD (Moving Average Convergence Divergence)
    
    MACD Line = 12-day EMA - 26-day EMA
    Signal Line = 9-day EMA of MACD Line
    Histogram = MACD Line - Signal Line
    
    Parameters:
    data: pandas Series of close prices
    fast_period: int, fast EMA period (default 12)
    slow_period: int, slow EMA period (default 26)  
    signal_period: int, signal line EMA period (default 9)
    
    Returns:
    dict with 'macd', 'signal', 'histogram' pandas Series
    
    Trading Signals:
    - MACD crosses above Signal = Bullish
    - MACD crosses below Signal = Bearish
    """
    ema_fast = calculate_ema(data, fast_period)
    ema_slow = calculate_ema(data, slow_period)
    
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    
    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }

def moving_average_crossover_signals(short_ma, long_ma):
    """
    Generate buy/sell signals based on moving average crossovers
    
    Parameters:
    short_ma: pandas Series of short-term moving average (e.g., 20-day)
    long_ma: pandas Series of long-term moving average (e.g., 50-day)
    
    Returns:
    pandas Series with signals: 1 for buy, -1 for sell, 0 for hold
    
    Strategy:
    - Golden Cross: Short MA crosses above Long MA = BUY
    - Death Cross: Short MA crosses below Long MA = SELL
    """
    signals = pd.Series(0, index=short_ma.index)
    
    # Buy signal: short MA crosses above long MA (Golden Cross)
    buy_signals = (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))
    
    # Sell signal: short MA crosses below long MA (Death Cross) 
    sell_signals = (short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))
    
    signals[buy_signals] = 1
    signals[sell_signals] = -1
    
    return signals

def identify_trend(short_ma, long_ma, price):
    """
    Identify current trend based on moving averages and price position
    
    Parameters:
    short_ma: pandas Series of short-term MA
    long_ma: pandas Series of long-term MA  
    price: pandas Series of current prices
    
    Returns:
    pandas Series with trend: 'UPTREND', 'DOWNTREND', 'SIDEWAYS'
    """
    trend = pd.Series('SIDEWAYS', index=price.index)
    
    # Uptrend: Price > Short MA > Long MA
    uptrend = (price > short_ma) & (short_ma > long_ma)
    
    # Downtrend: Price < Short MA < Long MA  
    downtrend = (price < short_ma) & (short_ma < long_ma)
    
    trend[uptrend] = 'UPTREND'
    trend[downtrend] = 'DOWNTREND'
    
    return trend
