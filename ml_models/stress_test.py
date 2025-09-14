# ml_models/stress_test.py - Test model under market stress
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_models.simple_price_predictor import SimplePricePredictor
from main_system import TechnicalAnalysisSystem

class MarketStressTester:
    def __init__(self):
        self.predictor = SimplePricePredictor()
    
    def simulate_market_crash(self, data, crash_day=500, crash_magnitude=-0.10):
        """Simulate a market crash and test model response"""
        print(f"💥 === SIMULATING MARKET CRASH ===")
        print(f"Crash Day: {crash_day}, Magnitude: {crash_magnitude*100:.1f}%")
        
        # Create synthetic crash
        data_crash = data.copy()
        crash_price = data_crash['Close'].iloc[crash_day]
        new_price = crash_price * (1 + crash_magnitude)
        
        # Simulate price gap and recovery
        for i in range(crash_day, min(crash_day + 10, len(data_crash))):
            recovery_factor = (i - crash_day) / 10  # Gradual recovery
            data_crash['Close'].iloc[i] = new_price + (crash_price - new_price) * recovery_factor
            data_crash['High'].iloc[i] = max(data_crash['High'].iloc[i], data_crash['Close'].iloc[i])
            data_crash['Low'].iloc[i] = min(data_crash['Low'].iloc[i], data_crash['Close'].iloc[i])
        
        return data_crash
    
    def test_prediction_under_stress(self, original_data):
        """Test how model performs during market stress"""
        print("🌪️ === STRESS TESTING YOUR MODEL ===")
        
        # Test different crash scenarios
        scenarios = [
            {"name": "Minor Correction", "magnitude": -0.05, "day": 400},
            {"name": "Major Correction", "magnitude": -0.10, "day": 600}, 
            {"name": "Market Crash", "magnitude": -0.20, "day": 800},
            {"name": "Circuit Breaker", "magnitude": -0.30, "day": 1000}
        ]
        
        results = []
        
        for scenario in scenarios:
            print(f"\n🎭 Testing: {scenario['name']}")
            
            # Create stressed data
            stressed_data = self.simulate_market_crash(
                original_data, 
                scenario['day'], 
                scenario['magnitude']
            )
            
            # Train model on stressed data
            self.predictor = SimplePricePredictor()
            model_results, best_model = self.predictor.train_models(stressed_data)
            
            # Test prediction accuracy around crash
            crash_day = scenario['day']
            test_range = range(max(0, crash_day-5), min(len(stressed_data), crash_day+10))
            
            predictions = []
            actuals = []
            
            for day in test_range:
                if day < len(stressed_data) - 1:
                    # Train on data up to this day
                    train_data = stressed_data.iloc[:day+1]
                    
                    if len(train_data) > 50:  # Minimum training data
                        try:
                            prediction = self.predictor.predict_tomorrow_price(train_data, best_model)
                            if prediction:
                                predictions.append(prediction['predicted_price'])
                                actuals.append(stressed_data['Close'].iloc[day+1])
                        except:
                            continue
            
            if predictions:
                pred_error = np.mean([abs(p-a)/a for p, a in zip(predictions, actuals)])
                scenario_result = {
                    'name': scenario['name'],
                    'error_rate': pred_error,
                    'r2_score': model_results[best_model]['r2']
                }
                results.append(scenario_result)
                
                print(f"   📊 R² Score: {model_results[best_model]['r2']:.4f}")
                print(f"   📉 Error Rate: {pred_error*100:.2f}%")
        
        return results
    
    def analyze_stress_results(self, results):
        """Analyze how model degrades under stress"""
        print(f"\n🎯 === STRESS TEST ANALYSIS ===")
        
        for result in results:
            print(f"\n🎭 {result['name']}:")
            print(f"   R² Score: {result['r2_score']:.4f}")
            print(f"   Error Rate: {result['error_rate']*100:.2f}%")
            
            if result['r2_score'] > 0.8:
                status = "✅ ROBUST"
            elif result['r2_score'] > 0.5:
                status = "⚠️ DEGRADED"
            else:
                status = "❌ FAILED"
            
            print(f"   Status: {status}")
        
        print(f"\n🎓 === STRESS TEST INSIGHTS ===")
        avg_performance = np.mean([r['r2_score'] for r in results])
        
        if avg_performance > 0.8:
            print("🏆 EXCELLENT: Model handles market stress well!")
        elif avg_performance > 0.6:
            print("✅ GOOD: Some degradation but still useful")
        elif avg_performance > 0.4:
            print("⚠️ CONCERNING: Significant performance loss under stress")
        else:
            print("❌ DANGEROUS: Model fails during market volatility")
        
        print(f"💡 Average Performance Under Stress: {avg_performance:.4f}")
        
        return avg_performance

if __name__ == "__main__":
    print("🌪️ === MARKET STRESS TESTING ===")
    print("Testing your model under extreme market conditions...")
    print("=" * 60)
    
    # Load data
    system = TechnicalAnalysisSystem()
    system.load_data()
    
    # Run stress tests
    tester = MarketStressTester()
    stress_results = tester.test_prediction_under_stress(system.data)
    
    # Analyze results
    if stress_results:
        avg_performance = tester.analyze_stress_results(stress_results)
        
        print(f"\n" + "="*60)
        print("🎓 STRESS TEST COMPLETE!")
        print(f"📊 Your model's resilience score: {avg_performance:.4f}")
        
        if avg_performance > 0.7:
            print("🚀 READY FOR LIVE TRADING!")
        else:
            print("🔧 NEEDS ROBUSTNESS IMPROVEMENTS")
    else:
        print("❌ Stress test failed - check data quality")
