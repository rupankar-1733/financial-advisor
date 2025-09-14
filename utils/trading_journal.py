# utils/trading_journal.py - Track your trading performance
import pandas as pd
from datetime import datetime
import os

class TradingJournal:
    def __init__(self):
        self.journal_file = "results/trading_journal.csv"
        self.setup_journal()
    
    def setup_journal(self):
        """Create journal file if it doesn't exist"""
        os.makedirs("results", exist_ok=True)
        
        if not os.path.exists(self.journal_file):
            columns = ['Date', 'Time', 'Action', 'Price', 'Quantity', 'Signal_Score', 
                      'RSI', 'Reason', 'Stop_Loss', 'Target', 'Status', 'PnL']
            df = pd.DataFrame(columns=columns)
            df.to_csv(self.journal_file, index=False)
            print(f"📓 Trading journal created: {self.journal_file}")
    
    def log_trade(self, action, price, quantity, signal_score, rsi, reason, stop_loss=None, target=None):
        """Log a new trade"""
        new_trade = {
            'Date': datetime.now().strftime("%Y-%m-%d"),
            'Time': datetime.now().strftime("%H:%M:%S"),
            'Action': action,
            'Price': price,
            'Quantity': quantity,
            'Signal_Score': signal_score,
            'RSI': rsi,
            'Reason': reason,
            'Stop_Loss': stop_loss,
            'Target': target,
            'Status': 'OPEN',
            'PnL': 0
        }
        
        df = pd.read_csv(self.journal_file)
        df = pd.concat([df, pd.DataFrame([new_trade])], ignore_index=True)
        df.to_csv(self.journal_file, index=False)
        
        print(f"📝 Trade logged: {action} {quantity} shares at ₹{price}")

# Usage example
if __name__ == "__main__":
    journal = TradingJournal()
    
    # Example: Log the BUY signal from today
    journal.log_trade(
        action="BUY",
        price=3133.85,
        quantity=10,
        signal_score=2.0,
        rsi=37.0,
        reason="Strong BUY signal at market close",
        stop_loss=3120.00,
        target=3150.00
    )
