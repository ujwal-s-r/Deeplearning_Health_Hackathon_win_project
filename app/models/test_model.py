"""
Test script to verify depression prediction model loading.
Run with: python -m app.models.test_model
"""

import sys
import os
import numpy as np
import tensorflow as tf

# Add parent directory to path to ensure imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from app.models.depression_predictor import DepressionPredictor, CustomAdamOptimizer

def test_model_loading():
    """Test loading the depression prediction model with different methods"""
    print("\n===== Testing Depression Prediction Model Loading =====")
    
    # Test paths
    model_path = os.path.join('app', 'models', 'depression_prediction_model.h5')
    alt_path = "C:\\Users\\ujwal\\OneDrive\\Desktop\\hack_1\\app\\models\\depression_prediction_model.h5"
    
    if os.path.exists(model_path):
        print(f"Model file found at: {model_path}")
    elif os.path.exists(alt_path):
        print(f"Model file found at: {alt_path}")
        model_path = alt_path
    else:
        print("WARNING: Model file not found at expected locations")
    
    # Create predictor with standard path
    print("\n--- Testing Default Loading Method ---")
    predictor = DepressionPredictor()
    print(f"Model loaded successfully: {predictor.loaded}")
    
    # If model loaded successfully, test prediction
    if predictor.loaded and predictor.model is not None:
        test_prediction(predictor)
    
def test_prediction(predictor):
    """Test making predictions with the loaded model"""
    print("\n--- Testing Prediction ---")
    
    # Create sample input data (12 features)
    sample_features = [
        0.45,   # blink_rate
        0.2,    # pupil_dilation_delta
        0.7,    # ratio_gaze_on_roi
        2,      # dominant_emotion (numeric code)
        8,      # phq8_score
        0.8,    # avg_reaction_time
        0.75,   # accuracy
        0.1,    # emotional_bias
        0.6,    # distraction_recovery
        0.5,    # distraction_response
        0.65,   # emotional_response_ratio
        0.8     # emoji_collection_ratio
    ]
    
    print(f"Sample features: {sample_features}")
    
    # Make prediction
    is_depressed, confidence = predictor.predict(sample_features)
    
    print(f"Prediction result:")
    print(f"  - Depression risk: {'High' if is_depressed else 'Low'}")
    print(f"  - Confidence: {confidence:.4f} ({confidence*100:.1f}%)")
    
    # Test prediction with different input values
    high_risk_features = sample_features.copy()
    high_risk_features[4] = 20  # Higher PHQ-8 score
    high_risk_features[1] = 0.05  # Lower pupil dilation
    
    is_depressed_high, confidence_high = predictor.predict(high_risk_features)
    
    print(f"\nPrediction with high-risk features:")
    print(f"  - Depression risk: {'High' if is_depressed_high else 'Low'}")
    print(f"  - Confidence: {confidence_high:.4f} ({confidence_high*100:.1f}%)")

def test_custom_optimizer():
    """Test the CustomAdamOptimizer class"""
    print("\n--- Testing CustomAdamOptimizer ---")
    
    # Create optimizer with weight_decay parameter
    try:
        optimizer = CustomAdamOptimizer(learning_rate=0.001, weight_decay=0.01)
        print("Successfully created CustomAdamOptimizer with weight_decay parameter")
        print(f"Optimizer learning rate: {optimizer.learning_rate.numpy()}")
    except Exception as e:
        print(f"Error creating CustomAdamOptimizer: {str(e)}")

def test_model_architecture():
    """Print the model architecture if loaded"""
    print("\n--- Model Architecture ---")
    
    predictor = DepressionPredictor()
    if predictor.loaded and predictor.model is not None:
        predictor.model.summary()
    else:
        print("Model not loaded, cannot display architecture")

if __name__ == "__main__":
    # Set TensorFlow logging level to reduce verbosity
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    print("Testing Depression Prediction Model\n")
    
    # Run tests
    test_model_loading()
    test_custom_optimizer()
    test_model_architecture()
    
    print("\nTest completed.") 