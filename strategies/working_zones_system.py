# strategies/working_zones_system.py - WORKING Zone Detection System
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class WorkingZoneDetector:
    def __init__(self, symbol, capital=10000):
        self.symbol = symbol
        self.capital = capital
        
    def get_price_zones(self, period='3mo'):
        """Get support and resistance zones using simple but effective methods"""
        print(f"🎯 === ZONE ANALYSIS: {self.symbol} ===")
        
        try:
            ticker = yf.Ticker(self.symbol)
            data = ticker.history(period=period, interval='1d')
            
            if len(data) < 20:
                print("❌ Insufficient data")
                return None
            
            current_price = data['Close'].iloc[-1]
            print(f"💰 Current Price: ₹{current_price:.2f}")
            print(f"📊 Data Points: {len(data)} days")
            
            # Method 1: Recent Highs and Lows (Most Reliable)
            recent_zones = self.find_recent_highs_lows(data, current_price)
            
            # Method 2: Moving Average Levels
            ma_zones = self.find_moving_average_levels(data, current_price)
            
            # Method 3: Volume-based levels
            volume_zones = self.find_high_volume_levels(data, current_price)
            
            # Method 4: Round number levels
            psychological_zones = self.find_psychological_levels(current_price)
            
            # Combine all zones
            all_zones = recent_zones + ma_zones + volume_zones + psychological_zones
            
            # Filter and rank zones
            filtered_zones = self.filter_and_rank_zones(all_zones, current_price)
            
            if not filtered_zones:
                print("❌ No valid zones found")
                return None
            
            # Separate support and resistance
            support_zones = [z for z in filtered_zones if z['price'] < current_price][:5]
            resistance_zones = [z for z in filtered_zones if z['price'] > current_price][:5]
            
            print(f"✅ Found {len(support_zones)} support zones, {len(resistance_zones)} resistance zones")
            
            return {
                'current_price': current_price,
                'support_zones': support_zones,
                'resistance_zones': resistance_zones,
                'all_zones': filtered_zones,
                'data': data
            }
            
        except Exception as e:
            print(f"❌ Error in zone detection: {e}")
            return None
    
    def find_recent_highs_lows(self, data, current_price):
        """Find recent swing highs and lows - most reliable method"""
        zones = []
        
        # Look for swing highs and lows in different time windows
        for window in [5, 10, 20]:
            for i in range(window, len(data) - window):
                high = data['High'].iloc[i]
                low = data['Low'].iloc[i]
                volume = data['Volume'].iloc[i]
                date = data.index[i]
                
                # Check if this is a swing high
                is_swing_high = True
                for j in range(i - window, i + window + 1):
                    if j != i and data['High'].iloc[j] >= high:
                        is_swing_high = False
                        break
                
                # Check if this is a swing low
                is_swing_low = True
                for j in range(i - window, i + window + 1):
                    if j != i and data['Low'].iloc[j] <= low:
                        is_swing_low = False
                        break
                
                # Add swing high as resistance zone
                if is_swing_high and abs(high - current_price) / current_price <= 0.15:  # Within 15%
                    zones.append({
                        'price': high,
                        'type': 'resistance',
                        'method': f'swing_high_{window}d',
                        'strength': 60 + window,
                        'volume': volume,
                        'date': date,
                        'age_days': (datetime.now() - date.tz_localize(None) if date.tz else (datetime.now() - date)).days
                    })
                
                # Add swing low as support zone
                if is_swing_low and abs(low - current_price) / current_price <= 0.15:  # Within 15%
                    zones.append({
                        'price': low,
                        'type': 'support',
                        'method': f'swing_low_{window}d',
                        'strength': 60 + window,
                        'volume': volume,
                        'date': date,
                        'age_days': (datetime.now() - date.tz_localize(None) if date.tz else (datetime.now() - date)).days
                    })
        
        return zones
    
    def find_moving_average_levels(self, data, current_price):
        """Find moving average support/resistance levels"""
        zones = []
        
        # Calculate different MAs
        for period in [20, 50, 100, 200]:
            if len(data) >= period:
                ma_series = data['Close'].rolling(period).mean()
                current_ma = ma_series.iloc[-1]
                
                if not pd.isna(current_ma):
                    zone_type = 'support' if current_ma < current_price else 'resistance'
                    
                    zones.append({
                        'price': current_ma,
                        'type': zone_type,
                        'method': f'MA_{period}',
                        'strength': 70 if period >= 200 else 60 if period >= 50 else 50,
                        'volume': 0,
                        'date': data.index[-1],
                        'age_days': 0
                    })
        
        return zones
    
    def find_high_volume_levels(self, data, current_price):
        """Find levels where high volume occurred"""
        zones = []
        
        # Find days with exceptionally high volume (top 20%)
        volume_threshold = data['Volume'].quantile(0.8)
        high_volume_days = data[data['Volume'] > volume_threshold].tail(20)  # Last 20 high volume days
        
        for idx, row in high_volume_days.iterrows():
            # Use the midpoint of high volume day as a potential zone
            mid_price = (row['High'] + row['Low']) / 2
            
            # Only include if within reasonable distance from current price
            if abs(mid_price - current_price) / current_price <= 0.12:  # Within 12%
                zone_type = 'support' if mid_price < current_price else 'resistance'
                
                zones.append({
                    'price': mid_price,
                    'type': zone_type,
                    'method': 'high_volume',
                    'strength': 55,
                    'volume': row['Volume'],
                    'date': idx,
                    'age_days': (datetime.now() - idx.tz_localize(None) if idx.tz else (datetime.now() - idx)).days
                })
        
        return zones
    
    def find_psychological_levels(self, current_price):
        """Find round number psychological levels"""
        zones = []
        
        # Determine step size based on price level
        if current_price > 2000:
            step = 100  # ₹2000, ₹2100, ₹2200, etc.
        elif current_price > 1000:
            step = 50   # ₹1000, ₹1050, ₹1100, etc.
        elif current_price > 500:
            step = 25   # ₹500, ₹525, ₹550, etc.
        else:
            step = 10   # ₹100, ₹110, ₹120, etc.
        
        # Find nearest round numbers above and below current price
        base_level = int(current_price / step) * step
        
        for i in range(-2, 3):  # 2 levels below to 2 levels above
            level = base_level + (i * step)
            if level > 0 and level != current_price:
                zone_type = 'support' if level < current_price else 'resistance'
                
                # Strength increases for rounder numbers
                strength = 45
                if level % (step * 2) == 0:  # Even rounder number
                    strength = 50
                if level % (step * 4) == 0:  # Very round number
                    strength = 55
                
                zones.append({
                    'price': level,
                    'type': zone_type,
                    'method': 'psychological',
                    'strength': strength,
                    'volume': 0,
                    'date': datetime.now(),
                    'age_days': 0
                })
        
        return zones
    
    def filter_and_rank_zones(self, zones, current_price):
        """Filter and rank zones by relevance and strength"""
        if not zones:
            return []
        
        # Remove duplicate levels (within 1% of each other)
        filtered = []
        for zone in zones:
            is_duplicate = False
            for existing in filtered:
                if abs(zone['price'] - existing['price']) / current_price < 0.01:  # Within 1%
                    # Keep the stronger one
                    if zone['strength'] > existing['strength']:
                        filtered.remove(existing)
                        filtered.append(zone)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(zone)
        
        # Calculate distance and final score
        for zone in filtered:
            distance_pct = abs(zone['price'] - current_price) / current_price * 100
            zone['distance_pct'] = distance_pct
            
            # Proximity bonus (closer zones are more relevant)
            proximity_score = max(0, 20 - distance_pct)  # Max 20 points for proximity
            
            # Age penalty (very old zones get slight penalty)
            age_penalty = min(zone['age_days'] / 30, 5)  # Max 5 point penalty
            
            # Final score
            zone['final_score'] = zone['strength'] + proximity_score - age_penalty
        
        # Sort by final score
        filtered.sort(key=lambda x: x['final_score'], reverse=True)
        
        return filtered[:15]  # Top 15 zones
    
    def generate_trading_plan(self, zones_data):
        """Generate specific trading plan"""
        if not zones_data:
            return None
        
        current_price = zones_data['current_price']
        support_zones = zones_data['support_zones']
        resistance_zones = zones_data['resistance_zones']
        
        shares_possible = int(self.capital / current_price)
        
        print(f"\n💰 === ₹{self.capital:,} TRADING PLAN ===")
        print(f"📊 Current Price: ₹{current_price:.2f}")
        print(f"📈 Max Shares: {shares_possible}")
        
        # Display support zones
        print(f"\n🔻 SUPPORT ZONES (BUY LEVELS):")
        print("-" * 85)
        print(f"{'LEVEL':<10} {'DISTANCE':<10} {'STRENGTH':<10} {'METHOD':<15} {'AGE'}")
        print("-" * 85)
        
        for zone in support_zones:
            level = f"₹{zone['price']:.0f}"
            distance = f"-{zone['distance_pct']:.1f}%"
            strength = f"{zone['final_score']:.0f}"
            method = zone['method']
            age = f"{zone['age_days']}d" if zone['age_days'] > 0 else "Current"
            
            print(f"{level:<10} {distance:<10} {strength:<10} {method:<15} {age}")
        
        # Display resistance zones
        print(f"\n🔺 RESISTANCE ZONES (SELL LEVELS):")
        print("-" * 85)
        print(f"{'LEVEL':<10} {'DISTANCE':<10} {'STRENGTH':<10} {'METHOD':<15} {'AGE'}")
        print("-" * 85)
        
        for zone in resistance_zones:
            level = f"₹{zone['price']:.0f}"
            distance = f"+{zone['distance_pct']:.1f}%"
            strength = f"{zone['final_score']:.0f}"
            method = zone['method']
            age = f"{zone['age_days']}d" if zone['age_days'] > 0 else "Current"
            
            print(f"{level:<10} {distance:<10} {strength:<10} {method:<15} {age}")
        
        # Generate specific recommendations
        self.generate_specific_recommendations(support_zones, resistance_zones, current_price, shares_possible)
        
        return {
            'support_zones': support_zones,
            'resistance_zones': resistance_zones,
            'shares_possible': shares_possible
        }
    
    def generate_specific_recommendations(self, support_zones, resistance_zones, current_price, shares_possible):
        """Generate specific trading recommendations"""
        print(f"\n🎯 === SPECIFIC TRADING RECOMMENDATIONS ===")
        
        if not support_zones or not resistance_zones:
            print("❌ Insufficient zone data for specific recommendations")
            return
        
        # Find best entry and exit zones
        best_support = support_zones[0]  # Highest scoring support
        best_resistance = resistance_zones[0]  # Highest scoring resistance
        
        entry_price = best_support['price']
        target_price = best_resistance['price']
        stop_loss_price = entry_price * 0.98  # 2% below entry
        
        # Calculate potential returns
        potential_profit = target_price - entry_price
        potential_loss = entry_price - stop_loss_price
        risk_reward_ratio = potential_profit / potential_loss if potential_loss > 0 else 0
        
        # Position sizing based on 2% risk rule
        risk_amount = self.capital * 0.02  # Risk 2% of capital
        position_size = min(int(risk_amount / potential_loss), shares_possible) if potential_loss > 0 else shares_possible
        
        print(f"💡 **STRATEGY**: Wait for pullback to support zone")
        print(f"📍 **Entry Zone**: ₹{entry_price:.0f} ({best_support['method']})")
        print(f"🎯 **Target Zone**: ₹{target_price:.0f} ({best_resistance['method']})")
        print(f"🛑 **Stop Loss**: ₹{stop_loss_price:.0f}")
        print(f"⚖️ **Risk:Reward**: 1:{risk_reward_ratio:.1f}")
        print(f"📊 **Position Size**: {position_size} shares (₹{position_size * entry_price:,.0f})")
        print(f"💰 **Max Risk**: ₹{potential_loss * position_size:,.0f}")
        
        # Expected returns
        expected_profit = potential_profit * position_size
        print(f"🎯 **Expected Profit**: ₹{expected_profit:,.0f}")
        print(f"📈 **ROI**: {(expected_profit / (position_size * entry_price)) * 100:.1f}%")
        
        # Trade quality assessment
        if risk_reward_ratio >= 3:
            quality = "🟢 EXCELLENT SETUP"
        elif risk_reward_ratio >= 2:
            quality = "🟡 GOOD SETUP"
        elif risk_reward_ratio >= 1.5:
            quality = "🟠 FAIR SETUP"
        else:
            quality = "🔴 POOR SETUP - AVOID"
        
        print(f"\n📊 **Trade Quality**: {quality}")
        
        # Action plan
        print(f"\n📋 **EXECUTION PLAN**:")
        print(f"1. 👀 Wait for price to drop to ₹{entry_price:.0f} zone")
        print(f"2. 📊 Confirm buying interest with increased volume")
        print(f"3. 🎯 Enter {position_size} shares at ₹{entry_price:.0f}")
        print(f"4. 🛑 Set stop loss at ₹{stop_loss_price:.0f}")
        print(f"5. 💰 Take profit at ₹{target_price:.0f}")
        print(f"6. 📱 Monitor overall market conditions")

def test_multiple_stocks():
    """Test zone detection on multiple stocks"""
    stocks = ['TCS.NS', 'INFY.NS', 'RELIANCE.NS', 'HDFCBANK.NS']
    
    for stock in stocks:
        print("\n" + "="*90)
        detector = WorkingZoneDetector(stock, capital=10000)
        
        zones_data = detector.get_price_zones()
        if zones_data:
            trading_plan = detector.generate_trading_plan(zones_data)
        else:
            print(f"❌ Could not analyze {stock}")

if __name__ == "__main__":
    print("🎯 === WORKING ZONE DETECTION SYSTEM ===")
    print("💡 Finding support/resistance zones for precision entries!")
    print("=" * 90)
    
    # Test the working zone detection system
    test_multiple_stocks()
    
    print("\n" + "="*90)
    print("✅ ZONE ANALYSIS COMPLETE!")
    print("🎯 History repeats - use these zones for precise entries and exits!")
