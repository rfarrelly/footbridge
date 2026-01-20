"""
Train ML predictor on historical data.
Run this once, then the trained model will be used automatically.
"""

import pandas as pd
from models.core import ModelConfig
from models.predictors_ml import MLPredictor


def main():
    print("Training ML Predictor...")

    # Load historical data
    df = pd.read_csv("historical_ppi_and_odds.csv")
    print(f"Loaded {len(df)} historical matches")

    # Initialize ML predictor
    config = ModelConfig()
    ml_predictor = MLPredictor(config, model_type="gradient_boosting")

    # Train
    print("Training...")
    metrics = ml_predictor.fit(df)

    # Print results
    print("\nTraining Results:")
    print(f"  CV AUC: {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}")
    print(f"  Test AUC: {metrics['test_auc']:.4f}")
    print(f"  Test Log Loss: {metrics['test_logloss']:.4f}")
    print(f"  Features: {metrics['n_features']}")

    # Save model
    import os

    os.makedirs("ml_models", exist_ok=True)
    ml_predictor.save_model("ml_models/ml_predictor.pkl")
    print("\nModel saved to ml_models/ml_predictor.pkl")
    print("ML predictor will now be used automatically in predictions!")


if __name__ == "__main__":
    main()
