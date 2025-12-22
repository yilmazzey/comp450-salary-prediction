import joblib
import numpy as np
from pathlib import Path

MODEL_PATH = Path("data/processed/best_model.joblib")

def verify():
    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully!")
        print(f"Model type: {type(model)}")
        
        # Test with a dummy input
        # Note: model expects a dataframe with 193 features
        # Just check if we can call something on it
        if hasattr(model, 'regressor_'):
            print("Regressor found in TransformedTargetRegressor.")
            
        print("NumPy version:", np.__version__)
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    verify()
