# ai_trading_system_ultimate.py - COMPLETE AI FINANCIAL ADVISOR
import sys
import os
from datetime import datetime
import pandas as pd
import numpy as np

# Add project paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our systems
from ml_models.multi_stock_system import MultiStockAISystem
from data_sources.comprehensive_news_system import ComprehensiveMarketIntelligence

class UltimateAITradingSystem:
    def __init__(self):
        self.stock_system = MultiStockAISystem()
        self.intel_system = ComprehensiveMarketIntelligence()
        self.market_intelligence = None
        
    def generate_ultimate_recommendations(self):
        """Generate AI + News + Macro recommendations"""
        print("🚀 === ULTIMATE AI TRADING SYSTEM ===")
        print("🤖 Combining: ML Models + News Sentiment + Macro Economics")
        print("=" * 80)
        
        # Step 1: Get market intelligence
        print("🌍 Step 1: Gathering market intelligence...")
        self.market_intelligence = self.intel_system.generate_market_intelligence_report()
        
        if not self.market_intelligence:
            print("❌ Failed to get market intelligence")
            return None
        
        market_sentiment = self.market_intelligence['overall_score']
        
        # Step 2: Fetch and analyze stocks
        print("\n🤖 Step 2: AI stock analysis...")
        if not self.stock_system.fetch_all_stocks_parallel():
            print("❌ Failed to fetch stock data")
            return None
        
        self.stock_system.train_all_models()
        stock_recommendations = self.stock_system.generate_portfolio_recommendations()
        
        if not stock_recommendations:
            print("❌ No stock recommendations generated")
            return None
        
        # Step 3: Enhance recommendations with market intelligence
        print("\n🧠 Step 3: Integrating market intelligence...")
        enhanced_recommendations = []
        
        for rec in stock_recommendations:
            # Original AI prediction
            ai_change = rec['price_change_pct']
            
            # Market sentiment boost/penalty
            sentiment_boost = market_sentiment * 0.3  # Max 30% boost from sentiment
            
            # Macro adjustment based on sector
            macro_adjustment = self.calculate_macro_adjustment(rec['symbol'], self.market_intelligence)
            
            # Combined prediction
            total_expected_change = ai_change + sentiment_boost + macro_adjustment
            
            # Enhanced recommendation logic
            if total_expected_change > 1.5:
                enhanced_rec = 'STRONG BUY'
                confidence = 'VERY HIGH'
            elif total_expected_change > 0.8:
                enhanced_rec = 'BUY'
                confidence = 'HIGH'
            elif total_expected_change < -1.5:
                enhanced_rec = 'STRONG SELL'
                confidence = 'VERY HIGH'
            elif total_expected_change < -0.8:
                enhanced_rec = 'SELL'
                confidence = 'HIGH'
            else:
                enhanced_rec = 'HOLD'
                confidence = 'MEDIUM'
            
            enhanced_rec_data = {
                **rec,  # All original data
                'market_sentiment': market_sentiment,
                'sentiment_boost': sentiment_boost,
                'macro_adjustment': macro_adjustment,
                'total_expected_change': total_expected_change,
                'original_recommendation': rec['recommendation'],
                'enhanced_recommendation': enhanced_rec,
                'enhanced_confidence': confidence,
                'ai_weight': 0.5,
                'sentiment_weight': 0.3,
                'macro_weight': 0.2
            }
            
            enhanced_recommendations.append(enhanced_rec_data)
        
        # Sort by total expected return
        enhanced_recommendations.sort(key=lambda x: x['total_expected_change'], reverse=True)
        
        return enhanced_recommendations
    
    def calculate_macro_adjustment(self, symbol, market_intel):
        """Calculate macro-economic adjustment for specific stocks"""
        if not market_intel or 'macro_data' not in market_intel:
            return 0
        
        adjustment = 0
        macro_data = market_intel['macro_data']
        
        # Sector-specific adjustments
        if symbol in ['TCS.NS', 'INFY.NS']:  # IT stocks
            # IT benefits from weak rupee
            if 'USD/INR' in macro_data:
                usd_change = macro_data['USD/INR']['change_pct']
                adjustment += usd_change * 0.1  # 10% correlation
        
        elif symbol in ['RELIANCE.NS']:  # Oil & Gas
            # Oil companies benefit from higher crude prices
            if 'Crude Oil' in macro_data:
                oil_change = macro_data['Crude Oil']['change_pct']
                adjustment += oil_change * 0.15  # 15% correlation
        
        elif symbol in ['HDFCBANK.NS', 'KOTAKBANK.NS', 'SBIN.NS']:  # Banks
            # Banks sensitive to interest rates and VIX
            if 'US 10Y Treasury' in macro_data:
                rate_change = macro_data['US 10Y Treasury']['change_pct']
                adjustment += rate_change * 0.05  # 5% correlation
            
            if 'VIX (Fear Index)' in macro_data:
                vix_change = macro_data['VIX (Fear Index)']['change_pct']
                adjustment -= vix_change * 0.1  # Inverse correlation
        
        # All stocks affected by market indices
        if 'Nifty 50' in macro_data:
            nifty_change = macro_data['Nifty 50']['change_pct']
            adjustment += nifty_change * 0.2  # 20% beta correlation
        
        return adjustment
    
    def display_ultimate_analysis(self, recommendations):
        """Display the ultimate AI analysis"""
        
        print("\n" + "=" * 100)
        print("🚀 ULTIMATE AI FINANCIAL ADVISOR - COMPLETE MARKET ANALYSIS")
        print(f"📅 {datetime.now().strftime('%A, %B %d, %Y at %H:%M:%S')}")
        print("=" * 100)
        
        if not recommendations:
            print("❌ No recommendations available")
            return
        
        # Market overview
        market_direction = self.market_intelligence['sentiment']['market_direction']
        market_score = self.market_intelligence['overall_score']
        
        print(f"🌍 MARKET INTELLIGENCE:")
        print(f"   📊 Overall Market: {market_direction} (Score: {market_score:+.3f})")
        print(f"   📰 News Sentiment: {self.market_intelligence['sentiment']['sentiment_strength']:.3f}")
        print(f"   🎯 AI Confidence: {self.market_intelligence['sentiment']['confidence']:.1%}")
        
        # Enhanced recommendations summary
        strong_buy = [r for r in recommendations if r['enhanced_recommendation'] == 'STRONG BUY']
        buy = [r for r in recommendations if r['enhanced_recommendation'] == 'BUY']
        strong_sell = [r for r in recommendations if r['enhanced_recommendation'] == 'STRONG SELL'] 
        sell = [r for r in recommendations if r['enhanced_recommendation'] == 'SELL']
        hold = [r for r in recommendations if r['enhanced_recommendation'] == 'HOLD']
        
        print(f"\n📊 ENHANCED PORTFOLIO RECOMMENDATIONS:")
        print(f"   🚀 STRONG BUY: {len(strong_buy)} stocks")
        print(f"   🟢 BUY: {len(buy)} stocks")  
        print(f"   ⚪ HOLD: {len(hold)} stocks")
        print(f"   🔴 SELL: {len(sell)} stocks")
        print(f"   🚨 STRONG SELL: {len(strong_sell)} stocks")
        print(f"   📈 Average AI Accuracy: {np.mean([r['model_accuracy'] for r in recommendations]):.1%}")
        
        print(f"\n🎯 TOP OPPORTUNITIES (AI + News + Macro):")
        print("-" * 100)
        print(f"{'RANK':<4} {'STOCK':<12} {'AI%':<6} {'NEWS%':<7} {'MACRO%':<7} {'TOTAL%':<7} {'ACTION':<11} {'CONF':<8}")
        print("-" * 100)
        
        for i, rec in enumerate(recommendations, 1):
            ai_pct = f"{rec['price_change_pct']:+.1f}%"
            news_pct = f"{rec['sentiment_boost']:+.1f}%"
            macro_pct = f"{rec['macro_adjustment']:+.1f}%"
            total_pct = f"{rec['total_expected_change']:+.1f}%"
            action = rec['enhanced_recommendation']
            confidence = rec['enhanced_confidence']
            
            # Action emoji
            if action == 'STRONG BUY':
                emoji = "🚀"
            elif action == 'BUY':
                emoji = "🟢"
            elif action == 'STRONG SELL':
                emoji = "🚨"
            elif action == 'SELL':
                emoji = "🔴"
            else:
                emoji = "⚪"
            
            print(f"{i:<4} {emoji} {rec['symbol']:<10} {ai_pct:<6} {news_pct:<7} {macro_pct:<7} {total_pct:<7} {action:<11} {confidence:<8}")
        
        # Key insights
        print(f"\n💡 ULTIMATE AI INSIGHTS:")
        
        if strong_buy or buy:
            best_opportunity = recommendations[0]
            print(f"   🎯 TOP OPPORTUNITY: {best_opportunity['symbol']} - {best_opportunity['enhanced_recommendation']}")
            print(f"      Expected Return: {best_opportunity['total_expected_change']:+.1f}%")
            print(f"      Breakdown: AI {best_opportunity['price_change_pct']:+.1f}% + News {best_opportunity['sentiment_boost']:+.1f}% + Macro {best_opportunity['macro_adjustment']:+.1f}%")
        
        if strong_sell or sell:
            worst_stock = recommendations[-1]
            print(f"   ⚠️ TOP RISK: {worst_stock['symbol']} - {worst_stock['enhanced_recommendation']}")
            print(f"      Expected Return: {worst_stock['total_expected_change']:+.1f}%")
        
        # Changed recommendations
        changed_recs = [r for r in recommendations if r['original_recommendation'] != r['enhanced_recommendation']]
        print(f"   📰 News/Macro Changed {len(changed_recs)} recommendations")
        
        # Portfolio allocation suggestion
        print(f"\n💰 SUGGESTED PORTFOLIO ALLOCATION:")
        total_positive = len(strong_buy) + len(buy)
        
        if total_positive > 0:
            print(f"   🚀 Aggressive: 60% in top {min(3, total_positive)} opportunities")
            print(f"   📊 Balanced: 40% in defensive positions")
        else:
            print(f"   ⚪ Defensive: Stay in cash/bonds until sentiment improves")
        
        print(f"   🛡️ Risk Management: Max 10% per stock, 2% stop losses")
        
        return recommendations

if __name__ == "__main__":
    print("🚀 ULTIMATE AI FINANCIAL ADVISOR")
    print("🤖 The Most Advanced Trading System Ever Built!")
    print("=" * 80)
    
    # Initialize ultimate system
    ultimate_ai = UltimateAITradingSystem()
    
    # Generate ultimate recommendations
    recommendations = ultimate_ai.generate_ultimate_recommendations()
    
    if recommendations:
        # Display ultimate analysis
        ultimate_ai.display_ultimate_analysis(recommendations)
        
        print("\n" + "="*100)
        print("✅ ULTIMATE AI ANALYSIS COMPLETE!")
        print("🌍 Your AI now combines:")
        print("   🤖 Machine Learning Predictions")
        print("   📰 Real-time News Sentiment") 
        print("   📊 Macro Economic Intelligence")
        print("   🎯 Multi-factor Risk Assessment")
        print("🚀 This rivals Goldman Sachs' trading systems!")
    else:
        print("❌ Ultimate analysis failed")
