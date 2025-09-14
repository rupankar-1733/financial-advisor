# automation/schedule_analysis.py
import schedule
import time
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main_system import TechnicalAnalysisSystem

def run_daily_analysis():
    print(f"\n🕐 Running scheduled analysis at {datetime.now()}")
    system = TechnicalAnalysisSystem()
    results = system.run_complete_analysis()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/daily_reports/analysis_{timestamp}.csv"
    
    if results is not None:
        results.to_csv(results_file)
        print(f"✅ Results saved to {results_file}")

# Schedule
schedule.every().day.at("09:30").do(run_daily_analysis)
schedule.every().day.at("15:30").do(run_daily_analysis)

if __name__ == "__main__":
    print("🔄 Analysis Scheduler Started...")
    while True:
        schedule.run_pending()
        time.sleep(60)
