# utils/live_data_fetcher.py - Real-time market data
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class LiveDataFetcher:
    def __init__(self, symbol="TCS.NS"):
        self.symbol = symbol
        
    def is_market_open(self):
        """Check if Indian market is currently open"""
        now = datetime.now()
        current_time = now.time()
        weekday = now.weekday()  # 0=Monday, 6=Sunday
        
        # Market hours: 9:15 AM to 3:30 PM, Monday to Friday
        market_open = datetime.strptime("09:15", "%H:%M").time()
        market_close = datetime.strptime("15:30", "%H:%M").time()
        
        is_weekday = weekday < 5  # Monday to Friday
        is_market_hours = market_open <= current_time <= market_close
        
        return is_weekday and is_market_hours
    
    def get_live_data(self, period="5d", interval="5m"):
        """Fetch live market data"""
        try:
            print(f"🔄 Fetching live data for {self.symbol}...")
            
            # Fetch recent data from yfinance
            ticker = yf.Ticker(self.symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                print(f"❌ No data received for {self.symbol}")
                return None
            
            # Clean column names
            data.columns = [col.title() for col in data.columns]
            
            # Ensure we have the required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                print(f"❌ Missing columns: {missing_columns}")
                return None
            
            print(f"✅ Live data fetched: {len(data)} records")
            print(f"📊 Data range: {data.index[0]} to {data.index[-1]}")
            print(f"💰 Current price: ₹{data['Close'].iloc[-1]:.2f}")
            
            return data[required_columns]
            
        except Exception as e:
            print(f"❌ Error fetching live data: {e}")
            return None
    
    def save_live_data(self, data, filename=None):
        """Save live data to file"""
        if data is None:
            return False
            
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/live_data_{timestamp}.csv"
        
        try:
            os.makedirs("data", exist_ok=True)
            data.to_csv(filename)
            print(f"✅ Live data saved to {filename}")
            return True
        except Exception as e:
            print(f"❌ Error saving data: {e}")
            return False

    def get_current_market_status(self):
        """Get detailed market status"""
        is_open = self.is_market_open()
        now = datetime.now()
        
        status = {
            'is_open': is_open,
            'current_time': now.strftime("%H:%M:%S"),
            'date': now.strftime("%Y-%m-%d"),
            'weekday': now.strftime("%A")
        }
        
        if is_open:
            market_close = datetime.combine(now.date(), datetime.strptime("15:30", "%H:%M").time())
            time_to_close = market_close - now
            status['time_to_close'] = str(time_to_close).split('.')[0]  # Remove microseconds
        else:
            # Next market open
            if now.weekday() >= 5:  # Weekend
                days_until_monday = 7 - now.weekday()
                next_open = datetime.combine(now.date(), datetime.strptime("09:15", "%H:%M").time()) + timedelta(days=days_until_monday)
            elif now.time() > datetime.strptime("15:30", "%H:%M").time():
                # After market close, next day
                next_open = datetime.combine(now.date() + timedelta(days=1), datetime.strptime("09:15", "%H:%M").time())
            else:
                # Before market open, same day
                next_open = datetime.combine(now.date(), datetime.strptime("09:15", "%H:%M").time())
            
            time_to_open = next_open - now
            status['time_to_open'] = str(time_to_open).split('.')[0]
        
        return status

if __name__ == "__main__":
    fetcher = LiveDataFetcher("TCS.NS")
    
    # Check market status
    status = fetcher.get_current_market_status()
    
    print("🏛️ === MARKET STATUS ===")
    print(f"Current Time: {status['current_time']} ({status['weekday']})")
    print(f"Market Open: {'✅ YES' if status['is_open'] else '❌ NO'}")
    
    if status['is_open']:
        print(f"⏰ Time to Close: {status['time_to_close']}")
        
        # Fetch live data
        live_data = fetcher.get_live_data()
        if live_data is not None:
            fetcher.save_live_data(live_data)
    else:
        print(f"⏰ Next Open: {status['time_to_open']}")
        print("💡 Using historical data for analysis")
