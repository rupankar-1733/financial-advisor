# ml_models/multi_stock_system_fixed.py - FIXED Multi-Stock AI System
import yfinance as yf
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import sys
import os
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_models.simple_price_predictor import SimplePricePredictor

class MultiStockAISystem:
    def __init__(self):
        # Top Indian stocks for portfolio
        self.stocks = {
            'TCS.NS': 'Tata Consultancy Services',
            'INFY.NS': 'Infosys',
            'RELIANCE.NS': 'Reliance Industries', 
            'HDFCBANK.NS': 'HDFC Bank',
            'ITC.NS': 'ITC Limited',
            'HINDUNILVR.NS': 'Hindustan Unilever',
            'SBIN.NS': 'State Bank of India',
            'BHARTIARTL.NS': 'Bharti Airtel',
            'KOTAKBANK.NS': 'Kotak Mahindra Bank',
            'LT.NS': 'Larsen & Toubro'
        }
        
        self.predictors = {}
        self.stock_data = {}
        
    def fetch_stock_data(self, symbol, period="1y", interval="1d"):
        """Fetch stock data - FIXED VERSION"""
        try:
            print(f"📊 Fetching {symbol} ({self.stocks[symbol]})...")
            ticker = yf.Ticker(symbol)
            
            # Use daily data instead of 5-minute (more reliable)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                print(f"❌ No data for {symbol}")
                return None
                
            # Clean data
            data = data.dropna()
            
            # Ensure we have required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_columns):
                print(f"❌ Missing columns for {symbol}")
                return None
            
            # Clean column names 
            data = data[required_columns]
            
            print(f"✅ {symbol}: {len(data)} days of data")
            return data
            
        except Exception as e:
            print(f"❌ Error fetching {symbol}: {e}")
            return None
    
    def fetch_all_stocks_parallel(self):
        """Fetch all stocks in parallel for speed"""
        print("🚀 === FETCHING MULTI-STOCK DATA ===")
        print("📅 Using daily data (more reliable than 5-minute)")
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_stock = {
                executor.submit(self.fetch_stock_data, symbol): symbol 
                for symbol in self.stocks.keys()
            }
            
            for future in future_to_stock:
                symbol = future_to_stock[future]
                try:
                    data = future.result()
                    if data is not None and len(data) > 50:
                        self.stock_data[symbol] = data
                    else:
                        print(f"❌ {symbol}: Insufficient data")
                except Exception as e:
                    print(f"❌ {symbol}: {e}")
        
        print(f"📈 Successfully loaded {len(self.stock_data)} stocks")
        return len(self.stock_data) > 0
    
    def adapt_predictor_for_daily_data(self, predictor):
        """Adapt our predictor for daily data instead of 5-minute"""
        # Modify the feature creation for daily data
        original_create_features = predictor.create_features
        
        def daily_create_features(data):
            """Modified feature creation for daily data"""
            print("🔧 Creating ML features for daily data...")
            df = data.copy()
            
            # LESSON 1: Technical indicators (adjusted for daily)
            print("   📊 Adding technical indicators...")
            df['sma_5'] = df['Close'].rolling(5).mean()
            df['sma_10'] = df['Close'].rolling(10).mean() 
            df['sma_20'] = df['Close'].rolling(20).mean()
            df['sma_50'] = df['Close'].rolling(50).mean()
            
            # LESSON 2: Price patterns
            print("   💰 Adding price patterns...")
            df['price_change'] = df['Close'].pct_change()
            df['high_low_ratio'] = df['High'] / df['Low']
            df['close_open_ratio'] = df['Close'] / df['Open']
            df['volatility'] = df['Close'].rolling(20).std()
            
            # LESSON 3: Volume features
            print("   📈 Adding volume features...")
            df['volume_sma_20'] = df['Volume'].rolling(20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
            
            # LESSON 4: Momentum indicators
            print("   ⚡ Adding momentum...")
            df['momentum_5'] = df['Close'] / df['Close'].shift(5)
            df['momentum_10'] = df['Close'] / df['Close'].shift(10)
            
            # LESSON 5: Lag features (yesterday's data)
            print("   ⏪ Adding historical data...")
            df['close_lag_1'] = df['Close'].shift(1)
            df['close_lag_2'] = df['Close'].shift(2)
            df['volume_lag_1'] = df['Volume'].shift(1)
            
            # LESSON 6: RSI for daily data
            print("   🎯 Adding RSI...")
            df['rsi'] = predictor.calculate_rsi(df['Close'], window=14)
            
            # LESSON 7: Target variable (what we want to predict)
            print("   🎯 Creating target (tomorrow's price)...")
            df['target'] = df['Close'].shift(-1)  # Next day's price
            
            # Remove rows with missing data
            df = df.dropna()
            
            # Feature selection (adjusted for daily data)
            feature_cols = ['sma_5', 'sma_10', 'sma_20', 'sma_50', 'price_change', 
                           'high_low_ratio', 'close_open_ratio', 'volatility', 'volume_ratio',
                           'momentum_5', 'momentum_10', 'close_lag_1', 'close_lag_2', 
                           'volume_lag_1', 'rsi']
            
            predictor.feature_columns = feature_cols
            
            print(f"   ✅ Created {len(feature_cols)} features from {len(df)} samples")
            return df[feature_cols + ['target']]
        
        # Replace the method
        predictor.create_features = daily_create_features
        return predictor
    
    def train_all_models(self):
        """Train prediction models for all stocks"""
        print("\n🤖 === TRAINING AI MODELS FOR ALL STOCKS ===")
        
        for symbol, data in self.stock_data.items():
            if len(data) > 100:  # Minimum data requirement
                print(f"\n🔄 Training model for {symbol} ({self.stocks[symbol]})...")
                
                predictor = SimplePricePredictor()
                predictor = self.adapt_predictor_for_daily_data(predictor)
                
                try:
                    results, best_model = predictor.train_models(data)
                    predictor.best_r2 = results[best_model]['r2']
                    predictor.best_model_name = best_model
                    
                    self.predictors[symbol] = predictor
                    
                    print(f"✅ {symbol}: R²={results[best_model]['r2']:.4f}, Error=₹{results[best_model]['rmse']:.2f}")
                    
                except Exception as e:
                    print(f"❌ {symbol}: Training failed - {e}")
            else:
                print(f"⚠️ {symbol}: Insufficient data ({len(data)} records)")
    
    def generate_portfolio_recommendations(self):
        """Generate AI recommendations for entire portfolio"""
        print("\n🎯 === AI PORTFOLIO RECOMMENDATIONS ===")
        
        recommendations = []
        
        for symbol, predictor in self.predictors.items():
            try:
                data = self.stock_data[symbol]
                prediction = predictor.predict_tomorrow_price(data, predictor.best_model_name)
                
                if prediction:
                    stock_name = self.stocks[symbol]
                    
                    rec = {
                        'symbol': symbol,
                        'name': stock_name,
                        'current_price': prediction['current_price'],
                        'predicted_price': prediction['predicted_price'],
                        'price_change_pct': prediction['price_change_pct'],
                        'direction': prediction['direction'],
                        'recommendation': prediction['recommendation'],
                        'confidence': prediction['confidence'],
                        'model_accuracy': predictor.best_r2
                    }
                    
                    recommendations.append(rec)
                    
            except Exception as e:
                print(f"❌ {symbol}: Prediction failed - {e}")
        
        # Sort by expected return (highest first)
        recommendations.sort(key=lambda x: x['price_change_pct'], reverse=True)
        
        return recommendations
    
    def display_portfolio_analysis(self, recommendations):
        """Display comprehensive portfolio analysis"""
        
        print("=" * 80)
        print("🚀 AI FINANCIAL ADVISOR - MULTI-STOCK PORTFOLIO ANALYSIS")
        print(f"📅 Analysis Date: {datetime.now().strftime('%A, %B %d, %Y at %H:%M:%S')}")
        print("=" * 80)
        
        if not recommendations:
            print("❌ No recommendations generated")
            return
        
        # Summary stats
        buy_signals = [r for r in recommendations if r['recommendation'] == 'BUY']
        sell_signals = [r for r in recommendations if r['recommendation'] == 'SELL']
        hold_signals = [r for r in recommendations if r['recommendation'] == 'HOLD']
        
        print(f"📊 Portfolio Overview:")
        print(f"   🟢 BUY Signals: {len(buy_signals)}")
        print(f"   🔴 SELL Signals: {len(sell_signals)}")
        print(f"   ⚪ HOLD Signals: {len(hold_signals)}")
        print(f"   📈 Average Model Accuracy: {np.mean([r['model_accuracy'] for r in recommendations]):.1%}")
        
        print(f"\n🎯 TOP OPPORTUNITIES:")
        print("-" * 80)
        print(f"{'RANK':<4} {'STOCK':<12} {'PRICE':<8} {'TARGET':<8} {'CHANGE':<8} {'ACTION':<6} {'ACCURACY':<8}")
        print("-" * 80)
        
        for i, rec in enumerate(recommendations, 1):
            current = f"₹{rec['current_price']:.0f}"
            predicted = f"₹{rec['predicted_price']:.0f}"
            change = f"{rec['price_change_pct']:+.2f}%"
            action = rec['recommendation']
            accuracy = f"{rec['model_accuracy']:.1%}"
            
            # Color coding
            if action == 'BUY':
                emoji = "🟢"
            elif action == 'SELL':
                emoji = "🔴"
            else:
                emoji = "⚪"
                
            print(f"{i:<4} {emoji} {rec['symbol']:<10} {current:<8} {predicted:<8} {change:<8} {action:<6} {accuracy:<8}")
        
        # Best opportunities
        print(f"\n💡 AI INSIGHTS:")
        
        if buy_signals:
            best_buy = max(buy_signals, key=lambda x: x['price_change_pct'])
            print(f"   🎯 BEST BUY: {best_buy['symbol']} - Expected +{best_buy['price_change_pct']:.2f}% ({best_buy['model_accuracy']:.1%} accuracy)")
            
        if sell_signals:
            best_sell = min(sell_signals, key=lambda x: x['price_change_pct'])
            print(f"   🎯 BEST SELL: {best_sell['symbol']} - Expected {best_sell['price_change_pct']:.2f}% ({best_sell['model_accuracy']:.1%} accuracy)")
        
        # Risk assessment
        high_confidence = [r for r in recommendations if r['model_accuracy'] > 0.8]
        print(f"   🎯 High Confidence Signals: {len(high_confidence)}/{len(recommendations)}")
        
        return recommendations

if __name__ == "__main__":
    print("🤖 AI FINANCIAL ADVISOR - MULTI-STOCK ANALYSIS")
    print(f"📅 {datetime.now().strftime('%A, %B %d, %Y at %H:%M:%S')}")
    print("🔄 Switching to daily data for better reliability...")
    
    # Initialize AI system
    ai_system = MultiStockAISystem()
    
    # Fetch data for all stocks
    if ai_system.fetch_all_stocks_parallel():
        # Train AI models
        ai_system.train_all_models()
        
        # Generate recommendations
        recommendations = ai_system.generate_portfolio_recommendations()
        
        if recommendations:
            # Display analysis
            ai_system.display_portfolio_analysis(recommendations)
            
            print("\n" + "="*80)
            print("✅ AI MULTI-STOCK ANALYSIS COMPLETE!")
            print("🚀 Your AI system successfully analyzed Indian stock market!")
        else:
            print("❌ No recommendations generated")
    else:
        print("❌ Failed to fetch stock data for any stocks")
