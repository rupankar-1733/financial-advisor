# ml_models/price_predictor.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class StockPricePredictor:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boost': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'linear_regression': LinearRegression(),
            'lstm': None  # Will be built dynamically
        }
        self.trained_models = {}
        self.feature_columns = []
        
    def create_features(self, data):
        """Create ML features from OHLCV data"""
        df = data.copy()
        
        # Technical indicators as features
        df['sma_5'] = df['Close'].rolling(5).mean()
        df['sma_10'] = df['Close'].rolling(10).mean()
        df['sma_20'] = df['Close'].rolling(20).mean()
        df['ema_12'] = df['Close'].ewm(span=12).mean()
        df['ema_26'] = df['Close'].ewm(span=26).mean()
        
        # Price-based features
        df['price_change'] = df['Close'].pct_change()
        df['high_low_ratio'] = df['High'] / df['Low']
        df['close_open_ratio'] = df['Close'] / df['Open']
        
        # Volume features
        df['volume_sma'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma']
        
        # Volatility features
        df['volatility'] = df['Close'].rolling(20).std()
        df['price_position'] = (df['Close'] - df['Low'].rolling(20).min()) / (df['High'].rolling(20).max() - df['Low'].rolling(20).min())
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            df[f'close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['Volume'].shift(lag)
        
        # Technical patterns
        df['rsi'] = self.calculate_rsi(df['Close'])
        macd_line = df['ema_12'] - df['ema_26']
        df['macd'] = macd_line
        df['macd_signal'] = macd_line.ewm(span=9).mean()
        
        # Target variable (next day's closing price)
        df['target'] = df['Close'].shift(-1)
        
        # Remove NaN values
        df = df.dropna()
        
        # Feature columns (exclude OHLCV and target)
        feature_cols = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'target']]
        self.feature_columns = feature_cols
        
        return df[feature_cols + ['target']]
    
    def calculate_rsi(self, prices, window=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def train_models(self, data, test_size=0.2):
        """Train all ML models"""
        print("🤖 === TRAINING PRICE PREDICTION MODELS ===")
        
        # Create features
        df_features = self.create_features(data)
        
        # Split data
        split_idx = int(len(df_features) * (1 - test_size))
        train_data = df_features.iloc[:split_idx]
        test_data = df_features.iloc[split_idx:]
        
        X_train = train_data[self.feature_columns]
        y_train = train_data['target']
        X_test = test_data[self.feature_columns]
        y_test = test_data['target']
        
        print(f"📊 Training on {len(X_train)} samples, testing on {len(X_test)} samples")
        print(f"📈 Features used: {len(self.feature_columns)}")
        
        results = {}
        
        # Train traditional ML models
        for name, model in self.models.items():
            if name != 'lstm':
                print(f"\n🔄 Training {name}...")
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results[name] = {
                    'mse': mse,
                    'rmse': np.sqrt(mse),
                    'r2': r2,
                    'model': model
                }
                
                print(f"   RMSE: {np.sqrt(mse):.4f}")
                print(f"   R²: {r2:.4f}")
                
                self.trained_models[name] = model
        
        # Train LSTM model
        print(f"\n🧠 Training LSTM Neural Network...")
        lstm_model = self.build_lstm_model(X_train.shape[1])
        
        # Reshape data for LSTM (samples, timesteps, features)
        X_train_lstm = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test_lstm = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))
        
        lstm_model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, verbose=0)
        y_pred_lstm = lstm_model.predict(X_test_lstm).flatten()
        
        mse_lstm = mean_squared_error(y_test, y_pred_lstm)
        r2_lstm = r2_score(y_test, y_pred_lstm)
        
        results['lstm'] = {
            'mse': mse_lstm,
            'rmse': np.sqrt(mse_lstm),
            'r2': r2_lstm,
            'model': lstm_model
        }
        
        print(f"   RMSE: {np.sqrt(mse_lstm):.4f}")
        print(f"   R²: {r2_lstm:.4f}")
        
        self.trained_models['lstm'] = lstm_model
        
        # Find best model
        best_model_name = min(results.keys(), key=lambda k: results[k]['rmse'])
        
        print(f"\n🏆 === MODEL PERFORMANCE SUMMARY ===")
        for name, metrics in results.items():
            status = "🥇 BEST" if name == best_model_name else "  "
            print(f"{status} {name.upper()}: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")
        
        # Save models
        os.makedirs("ml_models/saved_models", exist_ok=True)
        
        for name, model in self.trained_models.items():
            if name != 'lstm':
                joblib.dump(model, f"ml_models/saved_models/{name}_model.pkl")
            else:
                model.save(f"ml_models/saved_models/lstm_model.h5")
        
        print(f"\n✅ All models saved to ml_models/saved_models/")
        return results, best_model_name
    
    def build_lstm_model(self, n_features):
        """Build LSTM neural network"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(1, n_features)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def predict_next_price(self, current_data, model_name='random_forest'):
        """Predict next price using trained model"""
        if model_name not in self.trained_models:
            return None, "Model not trained"
        
        # Create features for current data
        df_features = self.create_features(current_data)
        
        if len(df_features) == 0:
            return None, "Insufficient data for prediction"
        
        # Get latest features
        latest_features = df_features[self.feature_columns].iloc[-1:].values
        
        model = self.trained_models[model_name]
        
        if model_name == 'lstm':
            latest_features = latest_features.reshape((1, 1, len(self.feature_columns)))
            prediction = model.predict(latest_features)[0][0]
        else:
            prediction = model.predict(latest_features)[0]
        
        current_price = current_data['Close'].iloc[-1]
        price_change = prediction - current_price
        price_change_pct = (price_change / current_price) * 100
        
        return {
            'predicted_price': prediction,
            'current_price': current_price,
            'price_change': price_change,
            'price_change_pct': price_change_pct,
            'direction': 'UP' if price_change > 0 else 'DOWN'
        }, None

# Test the model
if __name__ == "__main__":
    from main_system import TechnicalAnalysisSystem
    
    # Load data
    system = TechnicalAnalysisSystem()
    system.load_data()
    
    # Initialize and train predictor
    predictor = StockPricePredictor()
    results, best_model = predictor.train_models(system.data)
    
    # Make prediction
    prediction, error = predictor.predict_next_price(system.data, best_model)
    
    if prediction:
        print(f"\n🔮 === NEXT DAY PRICE PREDICTION ===")
        print(f"Current Price: ₹{prediction['current_price']:.2f}")
        print(f"Predicted Price: ₹{prediction['predicted_price']:.2f}")
        print(f"Expected Change: {prediction['price_change_pct']:.2f}% ({prediction['direction']})")
