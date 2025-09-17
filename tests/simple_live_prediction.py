# tests/simple_live_prediction.py - Quick live prediction test
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.live_data_fetcher import LiveDataFetcher
from datetime import datetime

def simple_prediction_test():
    print("🎯 === SIMPLE LIVE PREDICTION TEST ===")
    print(f"🕒 {datetime.now().strftime('%H:%M:%S')}")
    
    # Test stocks
    stocks = ['TCS.NS', 'INFY.NS', 'RELIANCE.NS']
    
    for stock in stocks:
        print(f"\n📊 {stock}:")
        
        fetcher = LiveDataFetcher(stock)
        
        # Check if we can get live data
        data = fetcher.get_live_data(period="5d", interval="1d")
        
        if data is not None and len(data) >= 5:
            # Simple analysis
            current = data['Close'].iloc[-1]
            prev = data['Close'].iloc[-2] 
            sma_5 = data['Close'].rolling(5).mean().iloc[-1]
            
            change = current - prev
            change_pct = (change / prev) * 100
            
            # Simple prediction logic
            if current > sma_5:
                signal = "🟢 BUY"
                confidence = "Medium"
            elif current < sma_5:
                signal = "🔴 SELL"  
                confidence = "Medium"
            else:
                signal = "⚪ HOLD"
                confidence = "Low"
            
            print(f"   💰 Price: ₹{current:.2f} ({change_pct:+.2f}%)")
            print(f"   📈 5-day avg: ₹{sma_5:.2f}")
            print(f"   🎯 Signal: {signal}")
            print(f"   💪 Confidence: {confidence}")
        else:
            print(f"   ❌ No data available")

if __name__ == "__main__":
    simple_prediction_test()
