# automation/weekend_prep.py - Weekend market preparation (FIXED)
import sys
import os
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from automation.smart_live_system import SmartLiveSystem

def weekend_market_prep():
    """Prepare for upcoming trading week"""
    print("📅 === WEEKEND MARKET PREPARATION ===")
    
    smart_system = SmartLiveSystem()
    status = smart_system.fetcher.get_current_market_status()
    
    if not status['is_open']:
        print(f"📊 Market closed - Next open: {status.get('time_to_open', 'Unknown')}")
        
        # Run historical analysis to prepare
        print("\n🔍 Running weekend analysis...")
        results = smart_system.run_intelligent_analysis()
        
        if results is not None:
            print("\n📋 === WEEK PREPARATION SUMMARY ===")
            
            # Get data from the smart system
            from main_system import TechnicalAnalysisSystem
            system = TechnicalAnalysisSystem(data_path=smart_system.historical_data_path)
            system.load_data()
            
            # Key levels to watch
            current = results.iloc[-1]
            current_price = system.data['Close'].iloc[-1]
            
            print(f"📊 Key Levels to Watch Monday:")
            print(f"   • Current Price: ₹{current_price:.2f}")
            print(f"   • Current RSI: {current['rsi']:.1f}")
            print(f"   • 20-day SMA: ₹{current['sma_20']:.2f} {'(Support)' if current_price > current['sma_20'] else '(Resistance)'}")
            print(f"   • 50-day SMA: ₹{current['sma_50']:.2f} {'(Support)' if current_price > current['sma_50'] else '(Resistance)'}")
            
            # Bollinger Bands levels
            print(f"   • BB Upper: ₹{current['bb_upper']:.2f}")
            print(f"   • BB Lower: ₹{current['bb_lower']:.2f}")
            
            # Recent signal pattern
            strong_signals = results[results['volume_confirmed'] != 0]
            if len(strong_signals) > 0:
                latest_signal = strong_signals.iloc[-1]
                signal_type = "BUY" if latest_signal['volume_confirmed'] > 0 else "SELL"
                signal_time = strong_signals.index[-1]
                print(f"   • Last Signal: {signal_type} at {signal_time} (Score: {latest_signal['volume_confirmed']})")
            
            print("\n🎯 === MONDAY TRADING STRATEGY ===")
            
            # RSI-based strategy
            if current['rsi'] < 35:
                print("🟢 RSI OVERSOLD SETUP:")
                print("   • Watch for reversal signals above ₹3135")
                print("   • Target: ₹3150-3160 range")
                print("   • Stop Loss: Below ₹3120")
            elif current['rsi'] > 65:
                print("🔴 RSI OVERBOUGHT SETUP:")
                print("   • Watch for selling opportunities below ₹3130") 
                print("   • Target: ₹3110-3120 range")
                print("   • Stop Loss: Above ₹3145")
            else:
                print("⚪ RSI NEUTRAL ZONE:")
                print("   • Wait for breakout above ₹3140 or below ₹3125")
                print("   • Volume confirmation required for any trades")
            
            # Trend analysis
            if current['sma_20'] > current['sma_50']:
                trend_status = "BULLISH BIAS"
                trend_emoji = "📈"
            else:
                trend_status = "BEARISH BIAS" 
                trend_emoji = "📉"
            
            print(f"\n{trend_emoji} Current Trend: {trend_status}")
            
            # Volume insights
            recent_volume = system.data['Volume'].tail(5).mean()
            avg_volume = current['volume_sma']
            
            if recent_volume > avg_volume:
                print("📊 Volume Status: ABOVE AVERAGE (Strong conviction)")
            else:
                print("📊 Volume Status: BELOW AVERAGE (Weak conviction)")
            
            # Risk assessment
            print(f"\n⚠️ === RISK ASSESSMENT ===")
            
            # Calculate volatility
            recent_prices = system.data['Close'].tail(20)
            volatility = recent_prices.std()
            price_range = recent_prices.max() - recent_prices.min()
            
            print(f"📊 20-day Volatility: {volatility:.2f}")
            print(f"📏 Recent Range: ₹{price_range:.2f}")
            
            if volatility < 5:
                risk_level = "LOW"
                risk_emoji = "✅"
            elif volatility < 10:
                risk_level = "MEDIUM"
                risk_emoji = "⚠️"
            else:
                risk_level = "HIGH"
                risk_emoji = "🚨"
                
            print(f"{risk_emoji} Risk Level: {risk_level}")
            
            # Final recommendations
            print(f"\n🎯 === MONDAY ACTION PLAN ===")
            print("1. 🕘 Monitor pre-market (9:00-9:15 AM) for gap movements")
            print("2. 📊 Wait for first 15 minutes to assess opening sentiment") 
            print("3. 🎯 Enter trades only with volume confirmation")
            print("4. ⛔ Use strict stop-losses (2% max risk per trade)")
            print("5. 📈 Take partial profits at key resistance levels")
            
            print(f"\n✅ System Status: READY FOR LIVE TRADING!")
            print("🔄 Run 'python automation/smart_live_system.py' Monday at 9:15 AM")

if __name__ == "__main__":
    weekend_market_prep()
