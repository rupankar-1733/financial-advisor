# ml_models/news_enhanced_system.py - AI + News Sentiment System
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from textblob import TextBlob
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_models.multi_stock_system import MultiStockAISystem

class NewsEnhancedAISystem(MultiStockAISystem):
    def __init__(self):
        super().__init__()
        self.news_sentiment = {}
    
    def fetch_stock_news(self, symbol):
        """Fetch news sentiment for a stock"""
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news[:5]  # Get latest 5 news items
            
            if not news:
                return {'sentiment': 0, 'count': 0}
            
            sentiments = []
            for item in news:
                title = item.get('title', '')
                summary = item.get('summary', '')
                text = f"{title} {summary}"
                
                if text.strip():
                    blob = TextBlob(text)
                    sentiments.append(blob.sentiment.polarity)
            
            avg_sentiment = np.mean(sentiments) if sentiments else 0
            
            return {
                'sentiment': avg_sentiment,
                'count': len(sentiments),
                'label': 'Positive' if avg_sentiment > 0.1 else 'Negative' if avg_sentiment < -0.1 else 'Neutral'
            }
            
        except Exception as e:
            return {'sentiment': 0, 'count': 0, 'label': 'Neutral'}
    
    def fetch_all_news_sentiment(self):
        """Get news sentiment for all stocks"""
        print("📰 === FETCHING NEWS SENTIMENT ===")
        
        for symbol in self.stocks.keys():
            if symbol in self.stock_data:
                print(f"📰 Analyzing news for {symbol}...")
                sentiment = self.fetch_stock_news(symbol)
                self.news_sentiment[symbol] = sentiment
                
                if sentiment['count'] > 0:
                    print(f"   {sentiment['label']} sentiment ({sentiment['sentiment']:.3f})")
                else:
                    print(f"   No recent news")
    
    def generate_enhanced_recommendations(self):
        """Generate AI + News recommendations"""
        print("\n🤖 === AI + NEWS ENHANCED RECOMMENDATIONS ===")
        
        recommendations = []
        
        for symbol, predictor in self.predictors.items():
            try:
                data = self.stock_data[symbol]
                prediction = predictor.predict_tomorrow_price(data, predictor.best_model_name)
                
                if prediction:
                    # Get news sentiment
                    news = self.news_sentiment.get(symbol, {'sentiment': 0, 'label': 'Neutral'})
                    
                    # Adjust recommendation based on news
                    original_rec = prediction['recommendation']
                    price_change = prediction['price_change_pct']
                    
                    # News sentiment boost/penalty
                    news_boost = news['sentiment'] * 0.5  # News can add up to 0.5% to expected change
                    enhanced_change = price_change + news_boost
                    
                    # Enhanced recommendation
                    if enhanced_change > 1:
                        enhanced_rec = 'BUY'
                    elif enhanced_change < -1:
                        enhanced_rec = 'SELL'
                    else:
                        enhanced_rec = 'HOLD'
                    
                    # Confidence adjustment
                    confidence = prediction['confidence']
                    if news['sentiment'] != 0 and abs(news['sentiment']) > 0.2:
                        confidence = 'VERY HIGH' if abs(enhanced_change) > 1 else 'HIGH'
                    
                    stock_name = self.stocks[symbol]
                    
                    rec = {
                        'symbol': symbol,
                        'name': stock_name,
                        'current_price': prediction['current_price'],
                        'predicted_price': prediction['predicted_price'],
                        'price_change_pct': price_change,
                        'news_sentiment': news['sentiment'],
                        'news_label': news['label'],
                        'enhanced_change': enhanced_change,
                        'original_rec': original_rec,
                        'enhanced_rec': enhanced_rec,
                        'confidence': confidence,
                        'model_accuracy': predictor.best_r2
                    }
                    
                    recommendations.append(rec)
                    
            except Exception as e:
                print(f"❌ {symbol}: Enhanced prediction failed - {e}")
        
        # Sort by enhanced expected return
        recommendations.sort(key=lambda x: x['enhanced_change'], reverse=True)
        
        return recommendations
    
    def display_enhanced_analysis(self, recommendations):
        """Display AI + News analysis"""
        
        print("=" * 90)
        print("🚀 AI + NEWS SENTIMENT FINANCIAL ADVISOR")
        print(f"📅 Analysis Date: {datetime.now().strftime('%A, %B %d, %Y at %H:%M:%S')}")
        print("=" * 90)
        
        if not recommendations:
            print("❌ No recommendations generated")
            return
        
        # Enhanced stats
        buy_signals = [r for r in recommendations if r['enhanced_rec'] == 'BUY']
        sell_signals = [r for r in recommendations if r['enhanced_rec'] == 'SELL']
        hold_signals = [r for r in recommendations if r['enhanced_rec'] == 'HOLD']
        
        positive_news = [r for r in recommendations if r['news_sentiment'] > 0.1]
        negative_news = [r for r in recommendations if r['news_sentiment'] < -0.1]
        
        print(f"📊 Enhanced Portfolio Overview:")
        print(f"   🟢 BUY Signals: {len(buy_signals)} (News Enhanced)")
        print(f"   🔴 SELL Signals: {len(sell_signals)} (News Enhanced)")
        print(f"   ⚪ HOLD Signals: {len(hold_signals)}")
        print(f"   📈 Average Model Accuracy: {np.mean([r['model_accuracy'] for r in recommendations]):.1%}")
        print(f"   📰 Positive News: {len(positive_news)} stocks")
        print(f"   📰 Negative News: {len(negative_news)} stocks")
        
        print(f"\n🎯 ENHANCED OPPORTUNITIES (AI + News):")
        print("-" * 90)
        print(f"{'RANK':<4} {'STOCK':<12} {'PRICE':<8} {'AI%':<6} {'NEWS':<8} {'TOTAL%':<7} {'ACTION':<6} {'ACC':<5}")
        print("-" * 90)
        
        for i, rec in enumerate(recommendations, 1):
            current = f"₹{rec['current_price']:.0f}"
            ai_change = f"{rec['price_change_pct']:+.1f}%"
            news_sentiment = rec['news_label'][:7]  # Truncate
            total_change = f"{rec['enhanced_change']:+.1f}%"
            action = rec['enhanced_rec']
            accuracy = f"{rec['model_accuracy']:.1%}"
            
            # Color coding
            if action == 'BUY':
                emoji = "🟢"
            elif action == 'SELL':
                emoji = "🔴"
            else:
                emoji = "⚪"
                
            print(f"{i:<4} {emoji} {rec['symbol']:<10} {current:<8} {ai_change:<6} {news_sentiment:<8} {total_change:<7} {action:<6} {accuracy:<5}")
        
        # Enhanced insights
        print(f"\n💡 AI + NEWS INSIGHTS:")
        
        if buy_signals:
            best_buy = max(buy_signals, key=lambda x: x['enhanced_change'])
            print(f"   🎯 BEST ENHANCED BUY: {best_buy['symbol']} - AI: {best_buy['price_change_pct']:+.1f}% + News: {best_buy['news_label']} = {best_buy['enhanced_change']:+.1f}%")
            
        if sell_signals:
            best_sell = min(sell_signals, key=lambda x: x['enhanced_change'])
            print(f"   🎯 BEST ENHANCED SELL: {best_sell['symbol']} - AI: {best_sell['price_change_pct']:+.1f}% + News: {best_sell['news_label']} = {best_sell['enhanced_change']:+.1f}%")
        
        # News impact analysis
        news_changed = [r for r in recommendations if r['original_rec'] != r['enhanced_rec']]
        print(f"   📰 News Changed Recommendations: {len(news_changed)} stocks")
        
        return recommendations

if __name__ == "__main__":
    print("🚀 AI + NEWS SENTIMENT FINANCIAL ADVISOR")
    print("🤖 Combining Machine Learning + Natural Language Processing")
    print("=" * 70)
    
    # Initialize enhanced AI system
    ai_system = NewsEnhancedAISystem()
    
    # Fetch stock data
    if ai_system.fetch_all_stocks_parallel():
        # Train AI models
        ai_system.train_all_models()
        
        # Fetch news sentiment
        ai_system.fetch_all_news_sentiment()
        
        # Generate enhanced recommendations
        recommendations = ai_system.generate_enhanced_recommendations()
        
        if recommendations:
            # Display enhanced analysis
            ai_system.display_enhanced_analysis(recommendations)
            
            print("\n" + "="*90)
            print("✅ AI + NEWS SENTIMENT ANALYSIS COMPLETE!")
            print("🚀 Your AI now combines technical analysis with market sentiment!")
        else:
            print("❌ No enhanced recommendations generated")
    else:
        print("❌ Failed to fetch stock data")
