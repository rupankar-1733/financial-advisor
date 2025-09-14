# automation/quick_commands.py - One-line commands
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main_system import TechnicalAnalysisSystem

def quick_status():
    """Quick market status"""
    system = TechnicalAnalysisSystem()
    system.load_data()
    signals = system.generate_trading_signals()
    
    if signals is not None:
        current = signals.iloc[-1]
        price = system.data['Close'].iloc[-1]
        
        print(f"💰 TCS: ₹{price:.2f}")
        print(f"📊 RSI: {current['rsi']:.1f}")
        print(f"🎯 Signal: {current['combined_signal']}")
        
        # Latest strong signal
        strong = signals[signals['volume_confirmed'] != 0]
        if len(strong) > 0:
            latest = strong.iloc[-1]
            signal_type = "BUY" if latest['volume_confirmed'] > 0 else "SELL"
            print(f"🔥 Latest: {signal_type} ({strong.index[-1]})")

def quick_signals():
    """Show only recent strong signals"""
    system = TechnicalAnalysisSystem()
    results = system.run_complete_analysis()
    
    if results is not None:
        strong = results[abs(results['volume_confirmed']) >= 2]
        print(f"📊 {len(strong)} Strong Signals Found:")
        for idx, row in strong.tail(10).iterrows():
            signal_type = "🟢 BUY" if row['volume_confirmed'] > 0 else "🔴 SELL"
            print(f"{idx.strftime('%m/%d %H:%M')}: {signal_type}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--status', action='store_true', help='Quick status')
    parser.add_argument('--signals', action='store_true', help='Recent signals')
    
    args = parser.parse_args()
    
    if args.status:
        quick_status()
    elif args.signals:
        quick_signals()
    else:
        print("Usage: python quick_commands.py --status  OR  --signals")
