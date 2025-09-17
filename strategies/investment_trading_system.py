# strategies/investment_trading_system.py - ₹10k Investment & Trading System
import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_sources.comprehensive_intelligence_system import UltimateMarketIntelligence

class InvestmentTradingSystem:
    def __init__(self, capital=10000):
        self.capital = capital
        self.intel_system = UltimateMarketIntelligence()
        
        # Investment vs Trading strategies
        self.investment_stocks = ['TCS.NS', 'RELIANCE.NS', 'HDFCBANK.NS', 'ITC.NS', 'HINDUNILVR.NS']
        self.trading_stocks = ['NIFTY', 'BANKNIFTY', 'TCS.NS', 'INFY.NS', 'RELIANCE.NS']
        
    def generate_investment_strategy(self):
        """Long-term investment strategy for ₹10k"""
        print("💰 === ₹10K INVESTMENT STRATEGY (3-12 months) ===")
        
        recommendations = []
        
        for stock in self.investment_stocks:
            zones = self.intel_system.calculate_support_resistance_zones(stock, period='6mo')
            
            if zones:
                current_price = zones['current_price']
                shares_possible = zones['shares_possible']
                
                # Investment scoring
                investment_score = self.calculate_investment_score(stock, zones)
                
                recommendation = {
                    'stock': stock.replace('.NS', ''),
                    'current_price': current_price,
                    'investment_amount': min(self.capital // len(self.investment_stocks), 2500),  # Max ₹2.5k per stock
                    'shares_to_buy': min(shares_possible, (self.capital // len(self.investment_stocks)) // current_price),
                    'target_price': zones['risk_levels']['take_profit_2'],
                    'stop_loss': zones['risk_levels']['stop_loss'],
                    'expected_return_pct': ((zones['risk_levels']['take_profit_2'] - current_price) / current_price) * 100,
                    'investment_score': investment_score,
                    'strategy': zones['entry_recommendation'],
                    'time_horizon': '3-12 months'
                }
                
                recommendations.append(recommendation)
        
        # Sort by investment score
        recommendations.sort(key=lambda x: x['investment_score'], reverse=True)
        
        print("\n📊 INVESTMENT RECOMMENDATIONS:")
        print("-" * 80)
        print(f"{'STOCK':<10} {'PRICE':<8} {'INVEST':<8} {'SHARES':<6} {'TARGET':<8} {'RETURN':<8} {'SCORE':<6}")
        print("-" * 80)
        
        total_investment = 0
        for rec in recommendations[:3]:  # Top 3 picks
            total_investment += rec['investment_amount']
            print(f"{rec['stock']:<10} ₹{rec['current_price']:<7.0f} "
                  f"₹{rec['investment_amount']:<7.0f} {rec['shares_to_buy']:<6.0f} "
                  f"₹{rec['target_price']:<7.0f} {rec['expected_return_pct']:<7.1f}% "
                  f"{rec['investment_score']:<6.1f}")
        
        print("-" * 80)
        print(f"Total Investment: ₹{total_investment:,}")
        print(f"Cash Remaining: ₹{self.capital - total_investment:,}")
        
        return recommendations[:3]
    
    def generate_trading_strategy(self):
        """Short-term trading strategy for ₹10k"""
        print("\n⚡ === ₹10K TRADING STRATEGY (1-30 days) ===")
        
        trading_opportunities = []
        
        for stock in self.trading_stocks:
            if stock in ['NIFTY', 'BANKNIFTY']:
                continue  # Skip indices for now
                
            zones = self.intel_system.calculate_support_resistance_zones(stock, period='1mo')
            
            if zones:
                trading_score = self.calculate_trading_score(stock, zones)
                
                opportunity = {
                    'stock': stock.replace('.NS', ''),
                    'current_price': zones['current_price'],
                    'position_size': min(5000, self.capital // 2),  # Max ₹5k per trade
                    'entry_zones': zones['support_zones'][:2],
                    'exit_zones': zones['resistance_zones'][:2],
                    'stop_loss': zones['risk_levels']['stop_loss'],
                    'risk_reward_ratio': self.calculate_risk_reward(zones),
                    'trading_score': trading_score,
                    'strategy_type': self.get_trading_strategy_type(zones),
                    'time_horizon': '1-30 days'
                }
                
                trading_opportunities.append(opportunity)
        
        # Sort by trading score
        trading_opportunities.sort(key=lambda x: x['trading_score'], reverse=True)
        
        print("\n⚡ TRADING OPPORTUNITIES:")
        print("-" * 90)
        print(f"{'STOCK':<10} {'PRICE':<8} {'POSITION':<9} {'ENTRY':<12} {'EXIT':<12} {'R:R':<6} {'SCORE':<6}")
        print("-" * 90)
        
        for opp in trading_opportunities[:2]:  # Top 2 trades
            entry_str = f"₹{opp['entry_zones'][0]:.0f}" if opp['entry_zones'] else "Market"
            exit_str = f"₹{opp['exit_zones'][0]:.0f}" if opp['exit_zones'] else "Target"
            
            print(f"{opp['stock']:<10} ₹{opp['current_price']:<7.0f} "
                  f"₹{opp['position_size']:<8.0f} {entry_str:<12} {exit_str:<12} "
                  f"{opp['risk_reward_ratio']:<6.1f} {opp['trading_score']:<6.1f}")
        
        return trading_opportunities[:2]
    
    def calculate_investment_score(self, stock, zones):
        """Calculate investment attractiveness score"""
        score = 50  # Base score
        
        current_price = zones['current_price']
        
        # Support/Resistance analysis
        if zones['support_zones']:
            support_distance = (current_price - zones['support_zones'][0]) / current_price * 100
            if support_distance < 5:  # Close to support
                score += 15
            elif support_distance < 10:
                score += 10
        
        if zones['resistance_zones']:
            resistance_distance = (zones['resistance_zones'][0] - current_price) / current_price * 100
            if resistance_distance > 15:  # Good upside potential
                score += 20
            elif resistance_distance > 10:
                score += 15
        
        # Risk-reward ratio
        if 'risk_levels' in zones:
            potential_gain = zones['risk_levels']['take_profit_2'] - current_price
            potential_loss = current_price - zones['risk_levels']['stop_loss']
            
            if potential_loss > 0:
                risk_reward = potential_gain / potential_loss
                if risk_reward > 3:
                    score += 20
                elif risk_reward > 2:
                    score += 15
                elif risk_reward > 1.5:
                    score += 10
        
        return min(score, 100)
    
    def calculate_trading_score(self, stock, zones):
        """Calculate short-term trading score"""
        score = 50  # Base score
        
        current_price = zones['current_price']
        
        # Volatility is good for trading
        if len(zones['support_zones']) > 1 and len(zones['resistance_zones']) > 1:
            score += 15  # Clear levels
        
        # Quick risk-reward opportunities
        if zones['resistance_zones'] and zones['support_zones']:
            upside = (zones['resistance_zones'][0] - current_price) / current_price * 100
            downside = (current_price - zones['support_zones'][0]) / current_price * 100
            
            if 3 <= upside <= 8 and downside <= 3:  # Good short-term setup
                score += 25
        
        return min(score, 100)
    
    def calculate_risk_reward(self, zones):
        """Calculate risk-reward ratio"""
        if not zones['resistance_zones'] or not zones['support_zones']:
            return 1.0
        
        current = zones['current_price']
        target = zones['resistance_zones'][0]
        stop = zones['support_zones'][0]
        
        potential_gain = target - current
        potential_loss = current - stop
        
        if potential_loss > 0:
            return potential_gain / potential_loss
        return 1.0
    
    def get_trading_strategy_type(self, zones):
        """Determine trading strategy type"""
        current = zones['current_price']
        
        if zones['support_zones']:
            support_distance = (current - zones['support_zones'][0]) / current * 100
            if support_distance < 2:
                return "BOUNCE_PLAY"
        
        if zones['resistance_zones']:
            resistance_distance = (zones['resistance_zones'][0] - current) / current * 100
            if resistance_distance < 2:
                return "BREAKDOWN_PLAY"
        
        return "MOMENTUM_PLAY"
    
    def display_complete_strategy(self):
        """Display both investment and trading strategies"""
        print("🚀 === COMPLETE ₹10K MARKET STRATEGY ===")
        print(f"💰 Total Capital: ₹{self.capital:,}")
        print(f"📅 Analysis Date: {datetime.now().strftime('%A, %B %d, %Y')}")
        
        # Generate strategies
        investment_recs = self.generate_investment_strategy()
        trading_opps = self.generate_trading_strategy()
        
        # Summary and allocation
        print("\n💡 === CAPITAL ALLOCATION RECOMMENDATION ===")
        print("👨‍💼 CONSERVATIVE (Investment Heavy):")
        print("   📊 70% Long-term Investment (₹7,000)")
        print("   ⚡ 30% Short-term Trading (₹3,000)")
        
        print("\n🎯 BALANCED (50-50):")
        print("   📊 50% Long-term Investment (₹5,000)")
        print("   ⚡ 50% Short-term Trading (₹5,000)")
        
        print("\n⚡ AGGRESSIVE (Trading Heavy):")
        print("   📊 30% Long-term Investment (₹3,000)")
        print("   ⚡ 70% Short-term Trading (₹7,000)")
        
        print("\n🎯 === NEXT STEPS ===")
        print("1. 📊 Monitor support/resistance levels daily")
        print("2. 📰 Track news and sentiment changes")
        print("3. 🔄 Adjust positions based on market regime")
        print("4. 📈 Review and rebalance weekly")
        
        return {
            'investment_recommendations': investment_recs,
            'trading_opportunities': trading_opps,
            'total_capital': self.capital
        }

if __name__ == "__main__":
    # Initialize system with ₹10k
    strategy_system = InvestmentTradingSystem(capital=10000)
    
    # Display complete strategy
    complete_strategy = strategy_system.display_complete_strategy()
    
    print("\n✅ COMPLETE STRATEGY GENERATED!")
    print("🚀 Ready for market action with ₹10,000!")
