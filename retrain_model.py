import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.compose import TransformedTargetRegressor

# Config
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "data/processed"
TRAIN_PATH = DATA_DIR / "so_2025_train.csv"
OUTPUT_MODEL_PATH = DATA_DIR / "best_model.joblib"
RANDOM_STATE = 450

def train():
    print(f"Loading data from {TRAIN_PATH}...")
    train_df = pd.read_csv(TRAIN_PATH)
    
    target_col = "CompYearlyUSD"
    feature_cols = [c for c in train_df.columns if c != target_col]
    
    X = train_df[feature_cols]
    y = train_df[target_col]
    
    print(f"Training on {X.shape[0]} samples with {X.shape[1]} features.")
    
    # Best model configuration found in 02_modeling.ipynb
    hgb = HistGradientBoostingRegressor(
        max_depth=12,
        learning_rate=0.05,
        max_iter=700,
        min_samples_leaf=20,
        random_state=RANDOM_STATE,
    )
    
    model = TransformedTargetRegressor(
        regressor=hgb,
        func=np.log1p,
        inverse_func=np.expm1,
    )
    
    print("Fitting model...")
    model.fit(X, y)
    
    print(f"Saving model to {OUTPUT_MODEL_PATH}...")
    joblib.dump(model, OUTPUT_MODEL_PATH)
    print("Retraining complete.")

if __name__ == "__main__":
    train()
