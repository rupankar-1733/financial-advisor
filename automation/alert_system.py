# automation/alert_system.py
import smtplib
import sys
import os
from email.mime.text import MIMEText
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main_system import TechnicalAnalysisSystem

class TradingAlerts:
    def __init__(self):
        # Email configuration (update with your details)
        self.sender_email = "your_email@gmail.com"
        self.password = "your_app_password" 
        self.receiver_email = "your_email@gmail.com"
    
    def send_alert(self, signal_info):
        """Send email alert for strong signals"""
        subject = f"🚨 TCS Alert: {signal_info['type']} Signal!"
        
        body = f"""
🚀 TCS TRADING ALERT

Signal: {signal_info['type']}
Strength: {signal_info['score']}
Price: ₹{signal_info['price']:.2f}
RSI: {signal_info['rsi']:.1f}
Time: {signal_info['time']}

⚠️ Automated alert - Do your research!
        """
        
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = self.sender_email
        msg['To'] = self.receiver_email
        
        try:
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(self.sender_email, self.password)
                server.send_message(msg)
            print("📧 Alert sent!")
        except Exception as e:
            print(f"❌ Alert failed: {e}")
    
    def monitor_signals(self):
        """Monitor for strong signals"""
        system = TechnicalAnalysisSystem()
        results = system.run_complete_analysis()
        
        if results is not None:
            latest = results.iloc[-1]
            if abs(latest['volume_confirmed']) >= 2:
                signal_info = {
                    'type': 'STRONG BUY' if latest['volume_confirmed'] > 0 else 'STRONG SELL',
                    'score': latest['volume_confirmed'],
                    'price': system.data['Close'].iloc[-1],
                    'rsi': latest['rsi'],
                    'time': results.index[-1]
                }
                self.send_alert(signal_info)

if __name__ == "__main__":
    alerts = TradingAlerts()
    alerts.monitor_signals()
