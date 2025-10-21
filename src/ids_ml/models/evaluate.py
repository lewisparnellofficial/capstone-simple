"""Model evaluation utilities."""

import json
import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder

from ids_ml.data.split import split_dataset
from ids_ml.models.inference import load_model


def evaluate_model(
    dataset_path="data/raw/dataset.csv", model_dir="models", output_file=None
):
    """
    Evaluate the trained model and report performance metrics.

    Args:
        dataset_path: Path to the dataset CSV file
        model_dir: Directory containing model artifacts
        output_file: Optional path to save metrics JSON

    Metrics reported:
    - Accuracy: Overall correctness of predictions
    - Precision: Proportion of positive predictions that are correct
    - Recall: Proportion of actual positives that are correctly identified
    - F1-Score: Harmonic mean of precision and recall

    Returns:
        Dictionary of evaluation metrics
    """

    print("Loading model and data...")

    try:
        model, label_mapping, scaler, selector, preprocessing_config = load_model(
            model_dir
        )
    except FileNotFoundError as e:
        print(f"Error: Model files not found in {model_dir}/")
        print(f"Please ensure the model has been trained first.")
        sys.exit(1)

    if preprocessing_config:
        print(f"Loaded preprocessing configuration:")
        print(
            f"  SMOTE: {'Enabled' if preprocessing_config.get('use_smote', False) else 'Disabled'}"
        )
        print(
            f"  Chi-Squared feature selection: {'Enabled' if preprocessing_config.get('use_chi2', False) else 'Disabled'}"
        )
        if preprocessing_config.get("use_chi2", False):
            print(
                f"  Chi-Squared k features: {preprocessing_config.get('chi2_k_features', 'N/A')}"
            )

    df = pd.read_csv(dataset_path)
    train_df, test_df, val_df = split_dataset(csv_path=dataset_path)

    le = LabelEncoder()
    le.fit(df["Label"])

    X_test = test_df.drop("Label", axis=1)
    y_test = le.transform(test_df["Label"])

    if scaler is not None and selector is not None:
        print("Applying preprocessing pipeline...")
        X_test_scaled = scaler.transform(X_test)
        X_test = selector.transform(X_test_scaled)
        print(f"Features after preprocessing: {X_test.shape[1]}")

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

    target_names_safe = [
        str(name).encode("ascii", "replace").decode("ascii") for name in target_names
    ]
    print(
        classification_report(
            y_test, y_pred, target_names=target_names_safe, zero_division=0
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

    if output_file is None:
        output_file = Path(model_dir) / "model_metrics.json"
    else:
        output_file = Path(output_file)

    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics saved to {output_file}")

    return metrics


def main():
    """CLI entry point for model evaluation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate trained IDS model on test dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  ids-evaluate

  ids-evaluate --dataset data/raw/custom.csv --model-dir trained_models/

  ids-evaluate --output results/evaluation_metrics.json
        """,
    )

    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="data/raw/dataset.csv",
        help="Path to dataset CSV file (default: data/raw/dataset.csv)",
    )

    parser.add_argument(
        "--model-dir",
        "-m",
        type=str,
        default="models",
        help="Directory containing model artifacts (default: models)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output path for metrics JSON (default: <model-dir>/model_metrics.json)",
    )

    args = parser.parse_args()

    print(f"\nEvaluation Configuration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Model directory: {args.model_dir}")
    print(
        f"  Output: {args.output if args.output else f'{args.model_dir}/model_metrics.json'}"
    )
    print()

    try:
        evaluate_model(
            dataset_path=args.dataset, model_dir=args.model_dir, output_file=args.output
        )
        return 0
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
