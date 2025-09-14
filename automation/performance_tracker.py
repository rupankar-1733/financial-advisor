# automation/performance_tracker.py
import pandas as pd
import sys
import os
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main_system import TechnicalAnalysisSystem

def analyze_signal_performance():
    """Analyze how profitable the signals have been"""
    print("📈 === SIGNAL PERFORMANCE ANALYSIS ===")
    
    system = TechnicalAnalysisSystem()
    results = system.run_complete_analysis()
    
    if results is None:
        return
    
    # Get all strong signals
    strong_signals = results[abs(results['volume_confirmed']) >= 2].copy()
    
    print(f"Total Strong Signals: {len(strong_signals)}")
    
    # Calculate theoretical returns
    data = system.data
    total_return = 0
    winning_trades = 0
    losing_trades = 0
    
    for i, (timestamp, signal) in enumerate(strong_signals.iterrows()):
        if i < len(strong_signals) - 1:  # Don't analyze the last signal
            entry_price = data.loc[timestamp, 'Close']
            
            # Find exit (next opposite signal or 5 periods later)
            future_signals = strong_signals.iloc[i+1:]
            
            if len(future_signals) > 0:
                exit_timestamp = future_signals.index[0]
                exit_price = data.loc[exit_timestamp, 'Close']
            else:
                # Use 5 periods later as exit
                try:
                    exit_idx = data.index.get_loc(timestamp) + 5
                    if exit_idx < len(data):
                        exit_price = data.iloc[exit_idx]['Close']
                    else:
                        continue
                except:
                    continue
            
            # Calculate return
            if signal['volume_confirmed'] > 0:  # BUY signal
                trade_return = (exit_price - entry_price) / entry_price * 100
            else:  # SELL signal
                trade_return = (entry_price - exit_price) / entry_price * 100
            
            total_return += trade_return
            
            if trade_return > 0:
                winning_trades += 1
                print(f"✅ {timestamp}: +{trade_return:.2f}% profit")
            else:
                losing_trades += 1
                print(f"❌ {timestamp}: {trade_return:.2f}% loss")
    
    total_trades = winning_trades + losing_trades
    
    if total_trades > 0:
        win_rate = winning_trades / total_trades * 100
        avg_return = total_return / total_trades
        
        print(f"\n📊 === PERFORMANCE SUMMARY ===")
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {winning_trades}")
        print(f"Losing Trades: {losing_trades}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Average Return per Trade: {avg_return:.2f}%")
        
        if win_rate > 60:
            print("🎯 EXCELLENT: High win rate system!")
        elif win_rate > 50:
            print("✅ GOOD: Profitable system")
        else:
            print("⚠️ REVIEW: Consider strategy adjustments")

if __name__ == "__main__":
    analyze_signal_performance()
