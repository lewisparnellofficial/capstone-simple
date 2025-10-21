"""Model inference utilities."""

import json
import pickle
from pathlib import Path

import pandas as pd
import xgboost as xgb


def load_model(model_dir="models"):
    """Load the best hyperparameters, the trained model, label mapping, and preprocessing objects.

    Args:
        model_dir: Directory containing model artifacts (default: 'models')

    Returns:
        tuple: (model, label_mapping, scaler, selector, preprocessing_config)
    """
    model_dir = Path(model_dir)

    with open(model_dir / "best_hyperparameters.json", "r") as f:
        hyperparameters = json.load(f)

    model = xgb.XGBClassifier(**hyperparameters)
    model.load_model(str(model_dir / "model.json"))

    with open(model_dir / "label_mapping.json", "r") as f:
        label_mapping = json.load(f)

    preprocessing_config = None
    if (model_dir / "preprocessing_config.json").exists():
        with open(model_dir / "preprocessing_config.json", "r") as f:
            preprocessing_config = json.load(f)

    scaler = None
    if (model_dir / "scaler.pkl").exists():
        with open(model_dir / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

    selector = None
    if (model_dir / "selector.pkl").exists():
        with open(model_dir / "selector.pkl", "rb") as f:
            selector = pickle.load(f)

    return model, label_mapping, scaler, selector, preprocessing_config


def predict(model, label_mapping, features, scaler=None, selector=None):
    """
    Make predictions on new data.

    Args:
        model: Loaded XGBoost model
        label_mapping: Dictionary mapping class indices to labels
        features: DataFrame or numpy array with the same features as training data
        scaler: Optional MinMaxScaler to apply before feature selection
        selector: Optional SelectKBest selector to apply after scaling

    Returns:
        predictions: Original label names
        probabilities: Probability scores for each class
    """

    if scaler is not None:
        features = scaler.transform(features)

    if selector is not None:
        features = selector.transform(features)

    y_pred_indices = model.predict(features)
    y_pred_proba = model.predict_proba(features)

    predictions = [label_mapping[str(int(idx))] for idx in y_pred_indices]

    return predictions, y_pred_proba
