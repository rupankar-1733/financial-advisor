# ml_models/overfitting_test.py - Test if your model is too good
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_models.simple_price_predictor import SimplePricePredictor
from main_system import TechnicalAnalysisSystem

def test_overfitting():
    """Test model on different time periods"""
    print("🔍 === OVERFITTING TEST ===")
    print("Testing model on different data splits...")
    
    system = TechnicalAnalysisSystem()
    system.load_data()
    
    # Test on multiple random splits
    test_results = []
    
    for i in range(5):
        print(f"\nTest {i+1}/5:")
        
        predictor = SimplePricePredictor()
        # Change random state to get different splits
        predictor.models['linear_regression'].random_state = i
        
        results, best_model = predictor.train_models(system.data)
        r2_score = results[best_model]['r2']
        rmse = results[best_model]['rmse']
        
        test_results.append({'r2': r2_score, 'rmse': rmse})
        print(f"   R²: {r2_score:.4f}, RMSE: ₹{rmse:.2f}")
    
    # Analyze consistency
    r2_scores = [r['r2'] for r in test_results]
    avg_r2 = sum(r2_scores) / len(r2_scores)
    r2_std = (sum((r - avg_r2)**2 for r in r2_scores) / len(r2_scores))**0.5
    
    print(f"\n📊 === STABILITY ANALYSIS ===")
    print(f"Average R²: {avg_r2:.4f}")
    print(f"R² Std Dev: {r2_std:.4f}")
    
    if r2_std < 0.01:
        print("✅ STABLE: Model performs consistently")
    elif r2_std < 0.05:
        print("⚠️ MODERATE: Some variation in performance")
    else:
        print("❌ UNSTABLE: High variation - possible overfitting")
    
    return avg_r2, r2_std

if __name__ == "__main__":
    test_overfitting()
