# automation/live_monitor.py - Continuous monitoring
import time
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main_system import TechnicalAnalysisSystem

class LiveMonitor:
    def __init__(self, check_interval=300):  # Check every 5 minutes
        self.check_interval = check_interval
        self.last_signal_time = None
        
    def monitor_continuously(self):
        print("🔄 Starting Live Monitor...")
        print(f"📊 Checking for new signals every {self.check_interval//60} minutes")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                print(f"🕐 Checking signals at {datetime.now().strftime('%H:%M:%S')}")
                
                system = TechnicalAnalysisSystem()
                results = system.run_complete_analysis()
                
                if results is not None:
                    # Check for new strong signals
                    latest_signals = results[results['volume_confirmed'] != 0]
                    
                    if len(latest_signals) > 0:
                        latest_signal = latest_signals.iloc[-1]
                        signal_time = latest_signals.index[-1]
                        
                        # Only alert if it's a new signal
                        if signal_time != self.last_signal_time:
                            signal_type = "🟢 BUY" if latest_signal['volume_confirmed'] > 0 else "🔴 SELL"
                            print(f"\n🚨 NEW SIGNAL DETECTED!")
                            print(f"   {signal_type} at {signal_time}")
                            print(f"   Score: {latest_signal['volume_confirmed']}")
                            print(f"   Price: ₹{system.data['Close'].iloc[-1]:.2f}")
                            
                            self.last_signal_time = signal_time
                
                print(f"✅ Check complete. Next check in {self.check_interval//60} minutes.\n")
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            print("\n🛑 Live monitor stopped")

if __name__ == "__main__":
    monitor = LiveMonitor(check_interval=300)  # 5 minutes
    monitor.monitor_continuously()
