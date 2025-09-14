# ml_models/simple_price_predictor.py - Complete ML Learning System
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class SimplePricePredictor:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boost': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'linear_regression': LinearRegression()
        }
        self.trained_models = {}
        self.feature_columns = []
        
    def create_features(self, data):
        """Create ML features - THIS IS WHERE YOU LEARN!"""
        print("🔧 Creating ML features...")
        df = data.copy()
        
        # LESSON 1: Technical indicators become ML features
        print("   📊 Adding technical indicators...")
        df['sma_5'] = df['Close'].rolling(5).mean()
        df['sma_10'] = df['Close'].rolling(10).mean() 
        df['sma_20'] = df['Close'].rolling(20).mean()
        
        # LESSON 2: Price patterns
        print("   💰 Adding price patterns...")
        df['price_change'] = df['Close'].pct_change()
        df['high_low_ratio'] = df['High'] / df['Low']
        df['volatility'] = df['Close'].rolling(10).std()
        
        # LESSON 3: Volume features
        print("   📈 Adding volume features...")
        df['volume_sma'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma']
        
        # LESSON 4: Lag features (yesterday's data)
        print("   ⏪ Adding historical data...")
        df['close_lag_1'] = df['Close'].shift(1)
        df['close_lag_2'] = df['Close'].shift(2)
        df['volume_lag_1'] = df['Volume'].shift(1)
        
        # LESSON 5: RSI as feature
        print("   🎯 Adding RSI...")
        df['rsi'] = self.calculate_rsi(df['Close'])
        
        # LESSON 6: Target variable (what we want to predict)
        print("   🎯 Creating target (tomorrow's price)...")
        df['target'] = df['Close'].shift(-1)  # Next day's price
        
        # Remove rows with missing data
        df = df.dropna()
        
        # Feature selection
        feature_cols = ['sma_5', 'sma_10', 'sma_20', 'price_change', 
                       'high_low_ratio', 'volatility', 'volume_ratio', 
                       'close_lag_1', 'close_lag_2', 'volume_lag_1', 'rsi']
        
        self.feature_columns = feature_cols
        
        print(f"   ✅ Created {len(feature_cols)} features from {len(df)} samples")
        return df[feature_cols + ['target']]
    
    def calculate_rsi(self, prices, window=14):
        """Calculate RSI for features"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def train_models(self, data):
        """Train models - LEARN HOW AI LEARNS!"""
        print("\n🤖 === TRAINING MACHINE LEARNING MODELS ===")
        
        # Step 1: Prepare data
        df_features = self.create_features(data)
        
        # Step 2: Split into features (X) and target (y)
        X = df_features[self.feature_columns]
        y = df_features['target']
        
        print(f"📊 Dataset: {len(X)} samples, {len(self.feature_columns)} features")
        
        # Step 3: Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"🎯 Training: {len(X_train)} samples, Testing: {len(X_test)} samples")
        
        results = {}
        
        # Step 4: Train each model
        for name, model in self.models.items():
            print(f"\n🔄 Training {name}...")
            
            # LESSON: This is where AI learns patterns!
            model.fit(X_train, y_train)
            
            # LESSON: Test the model's predictions
            y_pred = model.predict(X_test)
            
            # LESSON: Measure accuracy
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            results[name] = {
                'mse': mse,
                'rmse': np.sqrt(mse), 
                'mae': mae,
                'r2': r2,
                'model': model
            }
            
            # LESSON: Understand what these numbers mean
            print(f"   📈 R² Score: {r2:.4f} (Higher = Better, 1.0 = Perfect)")
            print(f"   📉 RMSE: ₹{np.sqrt(mse):.2f} (Lower = Better)")
            print(f"   📊 MAE: ₹{mae:.2f} (Average error)")
            
            self.trained_models[name] = model
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
        
        print(f"\n🏆 === MODEL COMPARISON ===")
        for name, metrics in results.items():
            status = "🥇 WINNER" if name == best_model_name else "  "
            print(f"{status} {name.upper()}: R²={metrics['r2']:.4f}, Error=₹{metrics['rmse']:.2f}")
        
        return results, best_model_name
    
    def explain_results(self, results, best_model_name):
        """TEACHER MODE: Explain what the results mean"""
        print(f"\n🎓 === LEARNING EXPLANATION ===")
        
        best_r2 = results[best_model_name]['r2']
        best_error = results[best_model_name]['rmse']
        
        print(f"🎯 Your best model: {best_model_name}")
        print(f"📊 R² Score: {best_r2:.4f}")
        
        if best_r2 > 0.8:
            print("   🎉 EXCELLENT! Model explains 80%+ of price movements")
        elif best_r2 > 0.6:
            print("   ✅ GOOD! Model captures most price patterns")
        elif best_r2 > 0.4:
            print("   ⚠️ FAIR! Some predictive power, needs improvement")
        else:
            print("   ❌ POOR! Model struggles to predict prices")
        
        print(f"💰 Average Error: ₹{best_error:.2f}")
        print(f"   💡 This means predictions are typically off by ₹{best_error:.2f}")
        
        print(f"\n🧠 What this means for trading:")
        if best_r2 > 0.6 and best_error < 10:
            print("   🚀 Strong enough for trading signals!")
        else:
            print("   🔧 Need more features or different approach")
    
    def predict_tomorrow_price(self, current_data, model_name='linear_regression'):
        """Make actual price prediction for tomorrow"""
        print(f"\n🔮 === PREDICTING TOMORROW'S PRICE ===")
        
        if model_name not in self.trained_models:
            return None
        
        # Create features for latest data
        df_features = self.create_features(current_data)
        
        if len(df_features) == 0:
            print("❌ Not enough data for prediction")
            return None
        
        # Get the very latest features (most recent day)
        latest_features = df_features[self.feature_columns].iloc[-1:].values
        
        # Make prediction
        model = self.trained_models[model_name]
        predicted_price = model.predict(latest_features)[0]
        
        # Current price for comparison
        current_price = current_data['Close'].iloc[-1]
        price_change = predicted_price - current_price
        price_change_pct = (price_change / current_price) * 100
        
        print(f"📊 Current Price: ₹{current_price:.2f}")
        print(f"🎯 Predicted Price: ₹{predicted_price:.2f}")
        print(f"📈 Expected Change: ₹{price_change:.2f} ({price_change_pct:+.2f}%)")
        
        if abs(price_change_pct) > 1:
            significance = "🚨 SIGNIFICANT"
        elif abs(price_change_pct) > 0.5:
            significance = "⚠️ MODERATE" 
        else:
            significance = "⚪ MINOR"
        
        direction = "🟢 UP" if price_change > 0 else "🔴 DOWN"
        
        print(f"🎪 Direction: {direction}")
        print(f"🎭 Significance: {significance}")
        
        # Trading recommendation
        action = "HOLD"  # Default
        confidence = "MEDIUM"
        
        if abs(price_change_pct) > 1:
            action = "BUY" if price_change > 0 else "SELL"
            confidence = "HIGH"
            print(f"🎯 AI Recommendation: {action}")
            print(f"💰 Confidence: {confidence} (R² = {self.best_r2:.1%})")
        else:
            print(f"🎯 AI Recommendation: {action}")
            print(f"💭 Reason: Small expected change ({price_change_pct:+.2f}%)")
        
        return {
            'current_price': current_price,
            'predicted_price': predicted_price,
            'price_change': price_change,
            'price_change_pct': price_change_pct,
            'direction': 'UP' if price_change > 0 else 'DOWN',
            'recommendation': action,
            'confidence': confidence,
            'significance': significance
        }
    
    def feature_importance_analysis(self, best_model_name):
        """ADVANCED: Show which features matter most"""
        print(f"\n🧠 === FEATURE IMPORTANCE ANALYSIS ===")
        
        model = self.trained_models[best_model_name]
        
        if hasattr(model, 'feature_importances_'):
            # For tree-based models
            importances = model.feature_importances_
            feature_importance = list(zip(self.feature_columns, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            print(f"🔍 Most Important Features for {best_model_name}:")
            for i, (feature, importance) in enumerate(feature_importance[:5]):
                bar = "█" * int(importance * 50)
                print(f"   {i+1}. {feature}: {importance:.3f} {bar}")
                
        elif hasattr(model, 'coef_'):
            # For linear models
            coefficients = abs(model.coef_)
            feature_importance = list(zip(self.feature_columns, coefficients))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            print(f"🔍 Most Important Features for {best_model_name}:")
            for i, (feature, coef) in enumerate(feature_importance[:5]):
                normalized_coef = coef / max(coefficients)
                bar = "█" * int(normalized_coef * 50)
                print(f"   {i+1}. {feature}: {coef:.3f} {bar}")
    
    def save_model(self, model_name, best_model_name):
        """Save the trained model"""
        if model_name in self.trained_models:
            os.makedirs("ml_models/saved_models", exist_ok=True)
            
            model_path = f"ml_models/saved_models/{model_name}_model.pkl"
            joblib.dump(self.trained_models[model_name], model_path)
            
            # Save feature columns
            features_path = f"ml_models/saved_models/{model_name}_features.pkl"
            joblib.dump(self.feature_columns, features_path)
            
            print(f"💾 Model saved: {model_path}")
            return True
        return False

if __name__ == "__main__":
    from main_system import TechnicalAnalysisSystem
    
    print("🚀 === COMPLETE ML STOCK PREDICTION SYSTEM ===")
    print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Load data
    system = TechnicalAnalysisSystem()
    system.load_data()
    
    # Create and train predictor
    predictor = SimplePricePredictor()
    results, best_model = predictor.train_models(system.data)
    
    # Store best R² for later use
    predictor.best_r2 = results[best_model]['r2']
    
    # Explain results
    predictor.explain_results(results, best_model)
    
    # Feature importance analysis
    predictor.feature_importance_analysis(best_model)
    
    # Make tomorrow's prediction
    print(f"\n" + "="*60)
    prediction = predictor.predict_tomorrow_price(system.data, best_model)
    
    if prediction:
        print(f"\n💡 === MONDAY TRADING STRATEGY ===")
        
        if prediction['recommendation'] != 'HOLD':
            print(f"🎯 Action: {prediction['recommendation']}")
            print(f"💰 Entry Price: ₹{prediction['current_price']:.2f}")
            print(f"🎪 Target Price: ₹{prediction['predicted_price']:.2f}")
            
            if prediction['recommendation'] == 'BUY':
                stop_loss = prediction['current_price'] * 0.98
                print(f"🛑 Stop Loss: ₹{stop_loss:.2f} (2% below entry)")
            else:
                stop_loss = prediction['current_price'] * 1.02
                print(f"🛑 Stop Loss: ₹{stop_loss:.2f} (2% above entry)")
                
        else:
            print("⚪ Strategy: HOLD and MONITOR")
            print("💭 Wait for more significant price movement signals")
    
    # Save the best model
    predictor.save_model(best_model, best_model)
    
    print(f"\n" + "="*60)
    print("✅ COMPLETE ML ANALYSIS FINISHED!")
    print("🚀 Your AI model is ready for Monday trading!")
