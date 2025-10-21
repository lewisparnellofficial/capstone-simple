import json

import pandas as pd
import xgboost as xgb


def load_model():
    """Load the best hyperparameters, the trained model, and label mapping."""
    with open("best_hyperparameters.json", "r") as f:
        hyperparameters = json.load(f)

    model = xgb.XGBClassifier(**hyperparameters)
    model.load_model("model.json")

    with open("label_mapping.json", "r") as f:
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


def main():
    model, label_mapping = load_model()

    print("Model loaded successfully!")
    print(f"Number of classes: {len(label_mapping)}")
    print(f"Classes: {list(label_mapping.values())}")

    df = pd.read_csv("dataset.csv")
    X_sample = df.drop("Label", axis=1).head(5)

    predictions, probabilities = predict(model, label_mapping, X_sample)

    print("\nPredictions:")
    for i, (pred, proba) in enumerate(zip(predictions, probabilities)):
        print(f"Sample {i + 1}: {pred} (confidence: {max(proba):.4f})")


if __name__ == "__main__":
    main()
