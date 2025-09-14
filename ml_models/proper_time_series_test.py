# ml_models/proper_time_series_test.py - CORRECT way to test time series
import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_models.simple_price_predictor import SimplePricePredictor
from main_system import TechnicalAnalysisSystem

class TimeSeriesValidator:
    def __init__(self):
        self.predictor = SimplePricePredictor()
    
    def time_series_split(self, data, n_splits=5):
        """PROPER time series cross validation"""
        print("🕒 === TIME SERIES VALIDATION (PROPER METHOD) ===")
        print("Using chronological splits - NO FUTURE DATA LEAKAGE!")
        
        # Create features
        df_features = self.predictor.create_features(data)
        X = df_features[self.predictor.feature_columns]
        y = df_features['target']
        
        n_samples = len(X)
        test_size = n_samples // (n_splits + 1)
        
        results = []
        
        for i in range(n_splits):
            # CRITICAL: Train on past, test on future
            train_end = (i + 1) * test_size + (n_samples - (n_splits + 1) * test_size)
            test_start = train_end
            test_end = test_start + test_size
            
            if test_end > n_samples:
                break
                
            X_train = X.iloc[:train_end]
            X_test = X.iloc[test_start:test_end]
            y_train = y.iloc[:train_end]
            y_test = y.iloc[test_start:test_end]
            
            print(f"\nFold {i+1}/{n_splits}:")
            print(f"   📊 Train: samples 0 to {train_end} (Past data)")
            print(f"   🔮 Test: samples {test_start} to {test_end} (Future data)")
            
            # Train model
            model = self.predictor.models['linear_regression']
            model.fit(X_train, y_train)
            
            # Test on future data
            y_pred = model.predict(X_test)
            
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            results.append({'r2': r2, 'rmse': rmse})
            
            print(f"   📈 R² Score: {r2:.4f}")
            print(f"   📉 RMSE: ₹{rmse:.2f}")
            
            if r2 < 0:
                print("   ⚠️ NEGATIVE R² - Model performs worse than random!")
        
        return results
    
    def analyze_real_performance(self, results):
        """Analyze realistic model performance"""
        if not results:
            return
            
        r2_scores = [r['r2'] for r in results]
        rmse_scores = [r['rmse'] for r in results]
        
        avg_r2 = np.mean(r2_scores)
        std_r2 = np.std(r2_scores)
        avg_rmse = np.mean(rmse_scores)
        
        print(f"\n📊 === REALISTIC PERFORMANCE ANALYSIS ===")
        print(f"Average R²: {avg_r2:.4f} (vs 0.9970 with data leakage)")
        print(f"R² Std Dev: {std_r2:.4f}")
        print(f"Average RMSE: ₹{avg_rmse:.2f}")
        
        print(f"\n🎓 === LEARNING INSIGHTS ===")
        
        if avg_r2 > 0.5:
            print("✅ GOOD: Model has real predictive power")
        elif avg_r2 > 0.2:
            print("⚠️ WEAK: Limited predictive ability")
        elif avg_r2 > 0:
            print("❌ POOR: Barely better than random")
        else:
            print("💀 TERRIBLE: Worse than random guessing!")
        
        if std_r2 > 0.2:
            print("📊 HIGH VARIANCE: Inconsistent performance")
        else:
            print("📊 STABLE: Consistent performance across time")
            
        # Reality check
        print(f"\n💡 === REALITY CHECK ===")
        print("🔍 Previous 99.7% accuracy was due to DATA LEAKAGE")
        print("✅ This is your model's REAL performance on unseen future data")
        print("📈 This is what you'd actually get in live trading")
        
        return avg_r2, std_r2, avg_rmse
    
    def walk_forward_validation(self, data):
        """Most realistic test - simulate live trading"""
        print(f"\n🚶 === WALK FORWARD VALIDATION ===")
        print("Simulating real trading: predict tomorrow, update model daily")
        
        df_features = self.predictor.create_features(data)
        X = df_features[self.predictor.feature_columns]
        y = df_features['target']
        
        # Start with minimum training data
        min_train_size = 200
        predictions = []
        actuals = []
        
        for i in range(min_train_size, len(X) - 1):
            # Train on all past data
            X_train = X.iloc[:i]
            y_train = y.iloc[:i]
            
            # Predict next day
            X_test = X.iloc[i:i+1]
            y_test = y.iloc[i]
            
            model = self.predictor.models['linear_regression']
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)[0]
            
            predictions.append(y_pred)
            actuals.append(y_test)
            
            if i % 50 == 0:  # Progress update
                print(f"   📅 Day {i}: Predicted ₹{y_pred:.2f}, Actual ₹{y_test:.2f}")
        
        # Calculate final performance
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        final_r2 = r2_score(actuals, predictions)
        final_rmse = np.sqrt(mean_squared_error(actuals, predictions))
        
        print(f"\n🎯 === WALK FORWARD RESULTS ===")
        print(f"R² Score: {final_r2:.4f}")
        print(f"RMSE: ₹{final_rmse:.2f}")
        print("💡 This is your MOST REALISTIC trading performance!")
        
        return final_r2, final_rmse

if __name__ == "__main__":
    print("🎓 PROPER ML VALIDATION FOR TIME SERIES")
    print("=" * 60)
    
    # Load data
    system = TechnicalAnalysisSystem()
    system.load_data()
    
    # Test properly
    validator = TimeSeriesValidator()
    
    # Time series cross validation
    results = validator.time_series_split(system.data, n_splits=5)
    real_performance = validator.analyze_real_performance(results)
    
    # Most realistic test
    walk_forward_results = validator.walk_forward_validation(system.data)
    
    print("\n" + "="*60)
    print("🎓 CONGRATULATIONS! You now understand:")
    print("✅ Data leakage problems")
    print("✅ Proper time series validation") 
    print("✅ Realistic ML performance expectations")
    print("✅ Why 99.7% accuracy was too good to be true!")
