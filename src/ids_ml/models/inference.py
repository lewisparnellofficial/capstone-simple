"""Model inference utilities."""

import json
from pathlib import Path

import pandas as pd
import xgboost as xgb


def load_model(model_dir="models"):
    """Load the best hyperparameters, the trained model, and label mapping.

    Args:
        model_dir: Directory containing model artifacts (default: 'models')

    Returns:
        tuple: (model, label_mapping)
    """
    model_dir = Path(model_dir)

    with open(model_dir / "best_hyperparameters.json", "r") as f:
        hyperparameters = json.load(f)

    model = xgb.XGBClassifier(**hyperparameters)
    model.load_model(str(model_dir / "model.json"))

    with open(model_dir / "label_mapping.json", "r") as f:
        label_mapping = json.load(f)

    return model, label_mapping


def predict(model, label_mapping, features):
    """
    Make predictions on new data.

    Args:
        model: Loaded XGBoost model
        label_mapping: Dictionary mapping class indices to labels
        features: DataFrame or numpy array with the same features as training data

    Returns:
        predictions: Original label names
        probabilities: Probability scores for each class
    """
    y_pred_indices = model.predict(features)
    y_pred_proba = model.predict_proba(features)

    predictions = [label_mapping[str(int(idx))] for idx in y_pred_indices]

    return predictions, y_pred_proba
