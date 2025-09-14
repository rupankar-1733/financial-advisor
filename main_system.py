# main_system.py - Complete TCS Technical Analysis System
import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

class TechnicalAnalysisSystem:
    """Complete Technical Analysis System"""
    
    def __init__(self, data_path=None):
        if data_path is None:
            self.data_path = os.path.join(PROJECT_ROOT, 'data', 'tcs_synthetic_5min.csv')
        else:
            self.data_path = data_path
        
        self.data = None
        self.signals = None
    
    def load_data(self):
        """Load market data"""
        try:
            self.data = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
            print(f"✅ Data loaded: {len(self.data)} records from {self.data.index.min()} to {self.data.index.max()}")
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            print(f"❌ Trying to load from: {self.data_path}")
            return False
    
    # === TECHNICAL INDICATORS ===
    def calculate_sma(self, data, window):
        return data.rolling(window=window).mean()

    def calculate_ema(self, data, window):
        return data.ewm(span=window, adjust=False).mean()

    def calculate_macd(self, data, fast_period=12, slow_period=26, signal_period=9):
        ema_fast = self.calculate_ema(data, fast_period)
        ema_slow = self.calculate_ema(data, slow_period)
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, signal_period)
        histogram = macd_line - signal_line
        return {'macd': macd_line, 'signal': signal_line, 'histogram': histogram}

    def calculate_rsi(self, data, window=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_bollinger_bands(self, data, window=20, num_std=2):
        sma = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return {'upper': upper_band, 'middle': sma, 'lower': lower_band}

    def calculate_volume_sma(self, volume, window=20):
        return volume.rolling(window=window).mean()

    def calculate_on_balance_volume(self, close, volume):
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

    def generate_trading_signals(self):
        """Complete trading signal generation"""
        if self.data is None:
            print("❌ No data loaded!")
            return None
            
        signals = pd.DataFrame(index=self.data.index)
        
        # Calculate all indicators
        signals['sma_20'] = self.calculate_sma(self.data['Close'], 20)
        signals['sma_50'] = self.calculate_sma(self.data['Close'], 50)
        signals['ema_12'] = self.calculate_ema(self.data['Close'], 12)
        signals['rsi'] = self.calculate_rsi(self.data['Close'])
        
        macd_data = self.calculate_macd(self.data['Close'])
        signals['macd'] = macd_data['macd']
        signals['macd_signal'] = macd_data['signal']
        
        bollinger = self.calculate_bollinger_bands(self.data['Close'])
        signals['bb_upper'] = bollinger['upper']
        signals['bb_lower'] = bollinger['lower']
        
        signals['volume_sma'] = self.calculate_volume_sma(self.data['Volume'])
        signals['obv'] = self.calculate_on_balance_volume(self.data['Close'], self.data['Volume'])
        
        # Generate buy/sell signals
        signals['ma_signal'] = 0
        signals['rsi_signal'] = 0
        signals['macd_signal_flag'] = 0
        signals['bb_signal'] = 0
        
        # Moving Average Crossover
        signals['ma_signal'] = np.where(
            (signals['sma_20'] > signals['sma_50']) & 
            (signals['sma_20'].shift(1) <= signals['sma_50'].shift(1)), 1,
            np.where(
                (signals['sma_20'] < signals['sma_50']) & 
                (signals['sma_20'].shift(1) >= signals['sma_50'].shift(1)), -1, 0
            )
        )
        
        # RSI Signals
        signals['rsi_signal'] = np.where(signals['rsi'] < 30, 1,
                                       np.where(signals['rsi'] > 70, -1, 0))
        
        # MACD Signals
        signals['macd_signal_flag'] = np.where(
            (signals['macd'] > signals['macd_signal']) & 
            (signals['macd'].shift(1) <= signals['macd_signal'].shift(1)), 1,
            np.where(
                (signals['macd'] < signals['macd_signal']) & 
                (signals['macd'].shift(1) >= signals['macd_signal'].shift(1)), -1, 0
            )
        )
        
        # Bollinger Band Signals
        signals['bb_signal'] = np.where(self.data['Close'] < signals['bb_lower'], 1,
                                      np.where(self.data['Close'] > signals['bb_upper'], -1, 0))
        
        # Combined Signal
        signals['combined_signal'] = (
            signals['ma_signal'] + 
            signals['rsi_signal'] + 
            signals['macd_signal_flag'] + 
            signals['bb_signal']
        )
        
        # Volume confirmation
        signals['volume_confirmed'] = np.where(
            (abs(signals['combined_signal']) >= 2) & 
            (self.data['Volume'] > signals['volume_sma']), 
            signals['combined_signal'], 0
        )
        
        self.signals = signals
        return signals

    def print_analysis(self):
        """Print complete analysis"""
        if self.signals is None:
            print("❌ No signals generated!")
            return
            
        print("🚀 === COMPLETE TECHNICAL ANALYSIS DASHBOARD ===")
        print(f"📊 Data Period: {self.data.index.min()} to {self.data.index.max()}")
        print(f"📈 Total Candles: {len(self.data)}")
        print()

        # Current Market State
        current = self.signals.iloc[-1]
        current_price = self.data['Close'].iloc[-1]

        print("💰 === CURRENT MARKET STATE ===")
        print(f"Current Price: ₹{current_price:.2f}")
        print(f"RSI: {current['rsi']:.1f} {'(OVERSOLD)' if current['rsi'] < 30 else '(OVERBOUGHT)' if current['rsi'] > 70 else '(NEUTRAL)'}")
        print(f"MACD: {current['macd']:.3f}")
        print(f"20-day SMA: ₹{current['sma_20']:.2f}")
        print(f"50-day SMA: ₹{current['sma_50']:.2f}")
        print()

        # Recent Signals
        print("🎯 === RECENT TRADING SIGNALS ===")
        recent_signals = self.signals.tail(20)
        strong_signals = recent_signals[abs(recent_signals['combined_signal']) >= 2]
        volume_confirmed = recent_signals[recent_signals['volume_confirmed'] != 0]

        print(f"Strong Signals (last 20 periods): {len(strong_signals)}")
        print(f"Volume-Confirmed Signals: {len(volume_confirmed)}")

        if len(volume_confirmed) > 0:
            print("\n🔥 LATEST VOLUME-CONFIRMED SIGNALS:")
            for idx, row in volume_confirmed.tail(3).iterrows():
                signal_type = "🟢 STRONG BUY" if row['volume_confirmed'] >= 2 else "🔴 STRONG SELL" if row['volume_confirmed'] <= -2 else "⚪ WEAK"
                print(f"{idx}: {signal_type} (Score: {row['volume_confirmed']})")

        print()
        print("📊 === INDICATOR SUMMARY ===")
        print(f"RSI Trend: {'Bullish' if current['rsi'] > 50 else 'Bearish'}")
        print(f"MA Trend: {'Bullish' if current['sma_20'] > current['sma_50'] else 'Bearish'}")
        print(f"Volume Status: {'Above Average' if self.data['Volume'].iloc[-1] > current['volume_sma'] else 'Below Average'}")
        print(f"Current Signal Score: {current['combined_signal']}")

    def run_complete_analysis(self):
        """Run the complete analysis"""
        print("🚀 Starting TCS Technical Analysis System...")
        print(f"📁 Project Root: {PROJECT_ROOT}")
        
        if not self.load_data():
            return None
        
        print("📊 Generating technical indicators...")
        signals = self.generate_trading_signals()
        
        if signals is not None:
            print("✅ Analysis complete! Displaying results...\n")
            self.print_analysis()
        
        return signals

# Main execution
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 TCS TECHNICAL ANALYSIS SYSTEM v1.0")
    print("=" * 60)
    
    # Initialize and run system
    system = TechnicalAnalysisSystem()
    results = system.run_complete_analysis()
    
    if results is not None:
        print("\n" + "=" * 60)
        print("✅ ANALYSIS COMPLETE!")
        print("=" * 60)
    else:
        print("\n❌ Analysis failed!")
