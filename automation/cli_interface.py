# automation/cli_interface.py
import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main_system import TechnicalAnalysisSystem

def main():
    parser = argparse.ArgumentParser(description='TCS Technical Analysis CLI')
    parser.add_argument('--signals-only', action='store_true', help='Show only signals')
    parser.add_argument('--export', help='Export to CSV file')
    parser.add_argument('--alert', action='store_true', help='Check for alerts')
    
    args = parser.parse_args()
    
    system = TechnicalAnalysisSystem()
    
    if args.alert:
        from automation.alert_system import TradingAlerts
        alerts = TradingAlerts()
        alerts.monitor_signals()
        return
    
    results = system.run_complete_analysis()
    
    if args.signals_only and results is not None:
        strong_signals = results[abs(results['volume_confirmed']) >= 2]
        print(f"\n📊 Strong Signals: {len(strong_signals)}")
        for idx, row in strong_signals.tail(5).iterrows():
            signal_type = "BUY" if row['volume_confirmed'] > 0 else "SELL"
            print(f"{idx}: {signal_type} (Score: {row['volume_confirmed']})")
    
    if args.export and results is not None:
        results.to_csv(f"results/{args.export}")
        print(f"✅ Exported to results/{args.export}")

if __name__ == "__main__":
    main()
