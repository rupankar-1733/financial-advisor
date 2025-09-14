# automation/market_hours_scheduler.py - Only run during market hours
import time
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from automation.smart_live_system import SmartLiveSystem

def market_hours_monitor():
    """Monitor only during market hours"""
    smart_system = SmartLiveSystem()
    
    print("📊 === MARKET HOURS MONITOR ===")
    print("🕘 Will run analysis every 5 minutes during market hours")
    print("💤 Will sleep when market is closed")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            status = smart_system.fetcher.get_current_market_status()
            
            if status['is_open']:
                print(f"🟢 Market Open - Running Analysis ({status['current_time']})")
                smart_system.run_intelligent_analysis()
                print("✅ Analysis complete. Next check in 5 minutes.\n")
                time.sleep(300)  # 5 minutes
            else:
                print(f"💤 Market Closed ({status['current_time']})")
                if 'time_to_open' in status:
                    print(f"⏰ Next open: {status['time_to_open']}")
                print("😴 Sleeping for 30 minutes...\n")
                time.sleep(1800)  # 30 minutes
                
    except KeyboardInterrupt:
        print("\n🛑 Market monitor stopped")

if __name__ == "__main__":
    market_hours_monitor()
