import json

import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder

from split_dataset import split_dataset


def evaluate_model():
    """
    Evaluate the trained model and report performance metrics.

    Metrics reported:
    - Accuracy: Overall correctness of predictions
    - Precision: Proportion of positive predictions that are correct
    - Recall: Proportion of actual positives that are correctly identified
    - F1-Score: Harmonic mean of precision and recall
    """

    print("Loading model and data...")

    with open("best_hyperparameters.json", "r") as f:
        hyperparameters = json.load(f)

    model = xgb.XGBClassifier(**hyperparameters)
    model.load_model("model.json")

    with open("label_mapping.json", "r") as f:
        label_mapping = json.load(f)

    df = pd.read_csv("dataset.csv")
    train_df, test_df, val_df = split_dataset()

    le = LabelEncoder()
    le.fit(df["Label"])

    X_test = test_df.drop("Label", axis=1)
    y_test = le.transform(test_df["Label"])

    print("Model loaded successfully!")
    print(f"Number of classes: {len(label_mapping)}")
    print(f"Test set size: {len(X_test):,} samples\n")

    print("Making predictions on test set...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    precision_weighted = precision_score(
        y_test, y_pred, average="weighted", zero_division=0
    )
    recall_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print("=" * 60)
    print("MODEL PERFORMANCE METRICS")
    print("=" * 60)
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("\n" + "-" * 60)
    print("WEIGHTED AVERAGES (accounts for class imbalance):")
    print("-" * 60)
    print(f"Precision: {precision_weighted:.4f}")
    print(f"Recall:    {recall_weighted:.4f}")
    print(f"F1-Score:  {f1_weighted:.4f}")

    print("\n" + "-" * 60)
    print("MACRO AVERAGES (treats all classes equally):")
    print("-" * 60)
    print(f"Precision: {precision_macro:.4f}")
    print(f"Recall:    {recall_macro:.4f}")
    print(f"F1-Score:  {f1_macro:.4f}")

    print("\n" + "=" * 60)
    print("DETAILED CLASSIFICATION REPORT")
    print("=" * 60)
    print()

    target_names = [label_mapping[str(i)] for i in range(len(label_mapping))]
    print(
        classification_report(
            y_test, y_pred, target_names=target_names, zero_division=0
        )
    )

    metrics = {
        "accuracy": float(accuracy),
        "weighted_metrics": {
            "precision": float(precision_weighted),
            "recall": float(recall_weighted),
            "f1_score": float(f1_weighted),
        },
        "macro_metrics": {
            "precision": float(precision_macro),
            "recall": float(recall_macro),
            "f1_score": float(f1_macro),
        },
        "test_set_size": len(X_test),
        "num_classes": len(label_mapping),
    }

    with open("model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nMetrics saved to model_metrics.json")

    return metrics


if __name__ == "__main__":
    evaluate_model()
