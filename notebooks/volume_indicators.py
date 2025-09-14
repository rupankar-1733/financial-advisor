import pandas as pd
import numpy as np

def calculate_volume_sma(volume, window=20):
    """
    Calculate Volume Simple Moving Average
    
    Used to identify above/below average volume periods
    
    Parameters:
    volume: pandas Series of volume data
    window: int, period for volume SMA
    
    Returns:
    pandas Series with volume SMA
    """
    return volume.rolling(window=window).mean()

def calculate_on_balance_volume(close, volume):
    """
    Calculate On-Balance Volume (OBV)
    
    OBV adds volume on up days and subtracts on down days
    - Rising OBV = Buying pressure
    - Falling OBV = Selling pressure
    
    Parameters:
    close: pandas Series of close prices
    volume: pandas Series of volume data
    
    Returns:
    pandas Series with OBV values
    """
    obv = pd.Series(index=close.index, dtype=float)
    obv.iloc[0] = volume.iloc[0]
    
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv

def identify_volume_spikes(volume, threshold_multiplier=2.0, window=20):
    """
    Identify volume spikes
    
    Volume spike = Current volume > (Average volume × threshold)
    
    Parameters:
    volume: pandas Series of volume data
    threshold_multiplier: float, multiplier for average volume
    window: int, period for average volume calculation
    
    Returns:
    pandas Series with boolean values (True = spike)
    """
    avg_volume = calculate_volume_sma(volume, window)
    return volume > (avg_volume * threshold_multiplier)

def calculate_volume_price_trend(close, volume):
    """
    Calculate Volume Price Trend (VPT)
    
    VPT = Previous VPT + [Volume × (Close - Previous Close) / Previous Close]
    
    Parameters:
    close: pandas Series of close prices
    volume: pandas Series of volume data
    
    Returns:
    pandas Series with VPT values
    """
    price_change_pct = close.pct_change()
    vpt = (volume * price_change_pct).cumsum()
    
    return vpt

def volume_confirmation_signals(price_signals, volume, avg_volume_window=20):
    """
    Confirm price signals with volume analysis
    
    Strong signals require above-average volume
    
    Parameters:
    price_signals: pandas Series with price-based signals (1, -1, 0)
    volume: pandas Series of volume data
    avg_volume_window: int, window for average volume
    
    Returns:
    pandas Series with volume-confirmed signals
    """
    avg_volume = calculate_volume_sma(volume, avg_volume_window)
    volume_confirmation = volume > avg_volume
    
    # Only keep signals when volume is above average
    confirmed_signals = price_signals.copy()
    confirmed_signals[~volume_confirmation] = 0
    
    return confirmed_signals
