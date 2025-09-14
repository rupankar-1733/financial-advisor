# automation/smart_live_system.py - Intelligent live/historical analysis
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main_system import TechnicalAnalysisSystem
from utils.live_data_fetcher import LiveDataFetcher

class SmartLiveSystem:
    def __init__(self):
        self.fetcher = LiveDataFetcher("TCS.NS")
        self.historical_data_path = "data/tcs_synthetic_5min.csv"
    
    def run_intelligent_analysis(self):
        """Run analysis with live data if market is open, historical if closed"""
        
        # Check market status
        status = self.fetcher.get_current_market_status()
        
        print("🚀 === SMART LIVE ANALYSIS SYSTEM ===")
        print(f"📅 {status['date']} {status['current_time']} ({status['weekday']})")
        
        if status['is_open']:
            print("🟢 MARKET IS OPEN - Using Live Data")
            print(f"⏰ Time to close: {status['time_to_close']}")
            
            # Try to get live data
            live_data = self.fetcher.get_live_data()
            
            if live_data is not None:
                # Save live data
                live_file = f"data/live_tcs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                live_data.to_csv(live_file)
                
                # Run analysis with live data
                system = TechnicalAnalysisSystem(data_path=live_file)
                results = system.run_complete_analysis()
                
                if results is not None:
                    print("\n🔥 === LIVE MARKET ANALYSIS ===")
                    self.display_live_insights(system, results)
                
                return results
            else:
                print("⚠️ Live data fetch failed - Using historical data")
        
        else:
            print("🔴 MARKET IS CLOSED")
            if 'time_to_open' in status:
                print(f"⏰ Next open: {status['time_to_open']}")
            
            print("📊 Running analysis with historical data...")
        
        # Fallback to historical data
        system = TechnicalAnalysisSystem(data_path=self.historical_data_path)
        results = system.run_complete_analysis()
        
        if results is not None:
            print("\n📈 === HISTORICAL DATA ANALYSIS ===")
            self.display_historical_insights(system, results)
        
        return results
    
    def display_live_insights(self, system, results):
        """Display insights for live market data"""
        current = results.iloc[-1]
        current_price = system.data['Close'].iloc[-1]
        
        print(f"💰 Live Price: ₹{current_price:.2f}")
        print(f"📊 Live RSI: {current['rsi']:.1f}")
        
        # Check for immediate signals
        if abs(current['combined_signal']) >= 2:
            signal_type = "🚨 STRONG BUY" if current['combined_signal'] > 0 else "🚨 STRONG SELL"
            print(f"{signal_type} SIGNAL ACTIVE!")
            print("⚡ IMMEDIATE ACTION RECOMMENDED")
        
        # Volume analysis
        if system.data['Volume'].iloc[-1] > current['volume_sma']:
            print("📈 HIGH VOLUME - Strong conviction")
        
        print(f"🎯 Signal Strength: {current['combined_signal']}")
    
    def display_historical_insights(self, system, results):
        """Display insights for historical data"""
        print("💡 This analysis is based on historical data")
        print("🔄 Will switch to live data when market opens")
        
        # Show latest strong signal
        strong_signals = results[results['volume_confirmed'] != 0]
        if len(strong_signals) > 0:
            latest = strong_signals.iloc[-1]
            signal_time = strong_signals.index[-1]
            signal_type = "BUY" if latest['volume_confirmed'] > 0 else "SELL"
            
            print(f"🔥 Last Strong Signal: {signal_type} at {signal_time}")
            print(f"📊 Signal Score: {latest['volume_confirmed']}")

if __name__ == "__main__":
    smart_system = SmartLiveSystem()
    smart_system.run_intelligent_analysis()
