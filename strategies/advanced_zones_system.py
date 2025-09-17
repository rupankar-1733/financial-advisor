# strategies/advanced_zones_system.py - Advanced Supply/Demand Zone Analysis
import sys
import os
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class AdvancedZoneAnalyzer:
    def __init__(self, symbol, capital=10000):
        self.symbol = symbol
        self.capital = capital
        self.zones = {}
        
    def identify_supply_demand_zones(self, period='3mo'):
        """Advanced supply/demand zone identification"""
        print(f"🎯 === ADVANCED ZONE ANALYSIS: {self.symbol} ===")
        
        ticker = yf.Ticker(self.symbol)
        data = ticker.history(period=period, interval='1d')
        
        if len(data) < 50:
            return None
            
        # Calculate volume-weighted zones
        self.zones = self.calculate_volume_zones(data)
        
        # Add psychological levels
        self.add_psychological_levels(data)
        
        # Add Fibonacci levels
        self.add_fibonacci_levels(data)
        
        # Rank zones by strength
        self.rank_zones_by_strength(data)
        
        return self.zones
    
    def calculate_volume_zones(self, data):
        """Identify zones based on volume and price action"""
        zones = {
            'demand_zones': [],
            'supply_zones': [],
            'current_price': data['Close'].iloc[-1],
            'analysis_date': datetime.now()
        }
        
        # Look for high volume rejection areas (supply/demand)
        volume_threshold = data['Volume'].quantile(0.8)  # Top 20% volume
        
        for i in range(10, len(data) - 10):
            current_vol = data['Volume'].iloc[i]
            current_high = data['High'].iloc[i]
            current_low = data['Low'].iloc[i]
            current_close = data['Close'].iloc[i]
            
            # Supply zone: High volume + rejection from highs
            if (current_vol > volume_threshold and 
                current_high == data['High'].iloc[i-5:i+5].max() and
                current_close < (current_high * 0.98)):  # Rejected from high
                
                supply_zone = {
                    'type': 'supply',
                    'level': current_high,
                    'zone_top': current_high * 1.01,
                    'zone_bottom': current_high * 0.99,
                    'volume': current_vol,
                    'date': data.index[i],
                    'strength': self.calculate_zone_strength(data, i, 'supply'),
                    'touches': 0,  # Will be calculated later
                    'status': 'untested'
                }
                zones['supply_zones'].append(supply_zone)
            
            # Demand zone: High volume + rejection from lows
            if (current_vol > volume_threshold and 
                current_low == data['Low'].iloc[i-5:i+5].min() and
                current_close > (current_low * 1.02)):  # Rejected from low
                
                demand_zone = {
                    'type': 'demand',
                    'level': current_low,
                    'zone_top': current_low * 1.01,
                    'zone_bottom': current_low * 0.99,
                    'volume': current_vol,
                    'date': data.index[i],
                    'strength': self.calculate_zone_strength(data, i, 'demand'),
                    'touches': 0,
                    'status': 'untested'
                }
                zones['demand_zones'].append(demand_zone)
        
        # Calculate how many times each zone was tested
        self.calculate_zone_touches(zones, data)
        
        return zones
    
    def calculate_zone_strength(self, data, index, zone_type):
        """Calculate the strength of a supply/demand zone"""
        strength = 50  # Base strength
        
        volume = data['Volume'].iloc[index]
        avg_volume = data['Volume'].iloc[max(0, index-20):index].mean()
        
        # Volume factor (higher volume = stronger zone)
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1
        strength += min(volume_ratio * 10, 25)
        
        # Time factor (older zones that hold = stronger)
        days_old = len(data) - index
        strength += min(days_old / 10, 15)
        
        # Wick length factor (longer rejection wick = stronger)
        if zone_type == 'supply':
            wick_length = (data['High'].iloc[index] - data['Close'].iloc[index]) / data['Close'].iloc[index]
        else:
            wick_length = (data['Close'].iloc[index] - data['Low'].iloc[index]) / data['Close'].iloc[index]
        
        strength += min(wick_length * 100, 10)
        
        return min(strength, 100)
    
    def calculate_zone_touches(self, zones, data):
        """Calculate how many times price touched each zone"""
        current_price = data['Close'].iloc[-1]
        
        # Count touches for supply zones
        for zone in zones['supply_zones']:
            touches = 0
            zone_date_idx = data.index.get_loc(zone['date'])
            
            # Check price action after zone creation
            for i in range(zone_date_idx + 1, len(data)):
                high = data['High'].iloc[i]
                low = data['Low'].iloc[i]
                
                # Price touched the zone
                if (low <= zone['zone_top'] and high >= zone['zone_bottom']):
                    touches += 1
            
            zone['touches'] = touches
            
            # Update zone status
            if touches == 0:
                zone['status'] = 'untested'
            elif touches <= 2:
                zone['status'] = 'strong'
            else:
                zone['status'] = 'weak'
        
        # Same for demand zones
        for zone in zones['demand_zones']:
            touches = 0
            zone_date_idx = data.index.get_loc(zone['date'])
            
            for i in range(zone_date_idx + 1, len(data)):
                high = data['High'].iloc[i]
                low = data['Low'].iloc[i]
                
                if (low <= zone['zone_top'] and high >= zone['zone_bottom']):
                    touches += 1
            
            zone['touches'] = touches
            
            if touches == 0:
                zone['status'] = 'untested'
            elif touches <= 2:
                zone['status'] = 'strong'
            else:
                zone['status'] = 'weak'
    
    def add_psychological_levels(self, data):
        """Add round number psychological levels"""
        current_price = data['Close'].iloc[-1]
        
        # Find nearest round numbers
        if current_price > 1000:
            step = 100
        elif current_price > 100:
            step = 50
        else:
            step = 10
        
        psychological_levels = []
        
        # Generate levels above and below current price
        for i in range(-3, 4):
            level = round(current_price / step) * step + (i * step)
            if level > 0:
                psychological_levels.append({
                    'type': 'psychological',
                    'level': level,
                    'strength': 70 if i == 0 else 60,  # Current level stronger
                    'description': f'Round number ₹{level}'
                })
        
        self.zones['psychological_levels'] = psychological_levels
    
    def add_fibonacci_levels(self, data):
        """Add Fibonacci retracement levels"""
        # Find recent swing high and low
        recent_data = data.tail(50)
        swing_high = recent_data['High'].max()
        swing_low = recent_data['Low'].min()
        
        fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        fibonacci_levels = []
        
        for fib in fib_levels:
            level = swing_low + (swing_high - swing_low) * fib
            fibonacci_levels.append({
                'type': 'fibonacci',
                'level': level,
                'percentage': fib,
                'strength': 65,
                'description': f'Fib {fib*100:.1f}% - ₹{level:.2f}'
            })
        
        self.zones['fibonacci_levels'] = fibonacci_levels
    
    def rank_zones_by_strength(self, data):
        """Rank all zones by strength and proximity"""
        current_price = self.zones['current_price']
        all_zones = []
        
        # Add supply zones
        for zone in self.zones.get('supply_zones', []):
            if zone['level'] > current_price:  # Only zones above current price
                distance = (zone['level'] - current_price) / current_price * 100
                zone['distance_pct'] = distance
                zone['proximity_score'] = max(0, 100 - distance * 2)  # Closer = higher score
                zone['total_score'] = (zone['strength'] + zone['proximity_score']) / 2
                all_zones.append(zone)
        
        # Add demand zones
        for zone in self.zones.get('demand_zones', []):
            if zone['level'] < current_price:  # Only zones below current price
                distance = (current_price - zone['level']) / current_price * 100
                zone['distance_pct'] = distance
                zone['proximity_score'] = max(0, 100 - distance * 2)
                zone['total_score'] = (zone['strength'] + zone['proximity_score']) / 2
                all_zones.append(zone)
        
        # Sort by total score
        all_zones.sort(key=lambda x: x['total_score'], reverse=True)
        self.zones['ranked_zones'] = all_zones
    
    def generate_trading_plan(self):
        """Generate specific trading plan for ₹10k"""
        print(f"\n💰 === ₹10K TRADING PLAN FOR {self.symbol} ===")
        
        if not self.zones:
            print("❌ No zones calculated")
            return None
        
        current_price = self.zones['current_price']
        shares_possible = self.capital // current_price
        
        # Find best demand zones (support)
        demand_zones = [z for z in self.zones.get('ranked_zones', []) if z['type'] == 'demand'][:3]
        
        # Find best supply zones (resistance)
        supply_zones = [z for z in self.zones.get('ranked_zones', []) if z['type'] == 'supply'][:3]
        
        print(f"📊 Current Price: ₹{current_price:.2f}")
        print(f"💵 Capital: ₹{self.capital:,}")
        print(f"📈 Shares Possible: {shares_possible}")
        
        print(f"\n🔻 TOP DEMAND ZONES (BUY LEVELS):")
        print("-" * 70)
        print(f"{'LEVEL':<10} {'DISTANCE':<10} {'STRENGTH':<10} {'STATUS':<10} {'TOUCHES'}")
        print("-" * 70)
        
        for i, zone in enumerate(demand_zones, 1):
            level = f"₹{zone['level']:.2f}"
            distance = f"-{zone['distance_pct']:.1f}%"
            strength = f"{zone['strength']:.0f}"
            status = zone['status'].title()
            touches = zone['touches']
            
            print(f"{level:<10} {distance:<10} {strength:<10} {status:<10} {touches}")
        
        print(f"\n🔺 TOP SUPPLY ZONES (SELL LEVELS):")
        print("-" * 70)
        print(f"{'LEVEL':<10} {'DISTANCE':<10} {'STRENGTH':<10} {'STATUS':<10} {'TOUCHES'}")
        print("-" * 70)
        
        for i, zone in enumerate(supply_zones, 1):
            level = f"₹{zone['level']:.2f}"
            distance = f"+{zone['distance_pct']:.1f}%"
            strength = f"{zone['strength']:.0f}"
            status = zone['status'].title()
            touches = zone['touches']
            
            print(f"{level:<10} {distance:<10} {strength:<10} {status:<10} {touches}")
        
        # Generate specific trade recommendation
        self.generate_trade_recommendation(demand_zones, supply_zones)
        
        return {
            'current_price': current_price,
            'demand_zones': demand_zones,
            'supply_zones': supply_zones,
            'shares_possible': shares_possible
        }
    
    def generate_trade_recommendation(self, demand_zones, supply_zones):
        """Generate specific trade recommendation"""
        print(f"\n🎯 === SPECIFIC TRADE RECOMMENDATION ===")
        
        current_price = self.zones['current_price']
        
        if not demand_zones or not supply_zones:
            print("❌ Insufficient zone data for recommendation")
            return
        
        best_demand = demand_zones[0]
        best_supply = supply_zones[0]
        
        # Calculate risk-reward
        entry_price = best_demand['level']
        target_price = best_supply['level']
        stop_loss = entry_price * 0.98  # 2% below entry
        
        potential_gain = target_price - entry_price
        potential_loss = entry_price - stop_loss
        risk_reward = potential_gain / potential_loss if potential_loss > 0 else 0
        
        # Position sizing based on risk
        risk_amount = self.capital * 0.02  # Risk 2% of capital
        shares_to_buy = min(risk_amount / potential_loss, self.capital // entry_price) if potential_loss > 0 else 0
        
        print(f"💡 STRATEGY: Wait for price to reach demand zone")
        print(f"📍 Entry Zone: ₹{entry_price:.2f} ({best_demand['distance_pct']:.1f}% below current)")
        print(f"🎯 Target Zone: ₹{target_price:.2f} ({(target_price/entry_price-1)*100:.1f}% profit)")
        print(f"🛑 Stop Loss: ₹{stop_loss:.2f}")
        print(f"⚖️ Risk:Reward = 1:{risk_reward:.1f}")
        print(f"📊 Position Size: {int(shares_to_buy)} shares (₹{int(shares_to_buy * entry_price):,})")
        print(f"💰 Total Risk: ₹{int(potential_loss * shares_to_buy):,} (2% of capital)")
        
        # Trade quality assessment
        if risk_reward > 3 and best_demand['status'] in ['untested', 'strong']:
            quality = "🟢 EXCELLENT TRADE SETUP"
        elif risk_reward > 2:
            quality = "🟡 GOOD TRADE SETUP"
        else:
            quality = "🔴 POOR RISK/REWARD - AVOID"
        
        print(f"\n📈 Trade Quality: {quality}")
        
        # Instructions
        print(f"\n📋 EXECUTION PLAN:")
        print(f"1. 👀 Watch for price to approach ₹{entry_price:.2f}")
        print(f"2. 📊 Confirm volume increase at demand zone")
        print(f"3. 🎯 Enter position with {int(shares_to_buy)} shares")
        print(f"4. 🛑 Set stop loss at ₹{stop_loss:.2f}")
        print(f"5. 💰 Take profit at ₹{target_price:.2f}")
        print(f"6. 📱 Monitor news and overall market sentiment")

def analyze_multiple_stocks():
    """Analyze zones for multiple stocks"""
    print("🎯 === MULTI-STOCK ZONE ANALYSIS ===")
    
    stocks = ['TCS.NS', 'INFY.NS', 'RELIANCE.NS', 'HDFCBANK.NS']
    
    for stock in stocks:
        print("\n" + "="*80)
        analyzer = AdvancedZoneAnalyzer(stock)
        zones = analyzer.identify_supply_demand_zones()
        
        if zones:
            trading_plan = analyzer.generate_trading_plan()

if __name__ == "__main__":
    # Analyze TCS in detail
    tcs_analyzer = AdvancedZoneAnalyzer('TCS.NS')
    tcs_zones = tcs_analyzer.identify_supply_demand_zones()
    
    if tcs_zones:
        tcs_plan = tcs_analyzer.generate_trading_plan()
    
    # Quick analysis of other stocks
    analyze_multiple_stocks()
    
    print("\n" + "="*80)
    print("✅ ADVANCED ZONE ANALYSIS COMPLETE!")
    print("🎯 Ready for precision trading with zone-based entries!")
