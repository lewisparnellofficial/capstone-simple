"""Model training."""

import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

from ids_ml.models.utils import (
    get_base_params,
    get_default_hyperparameters,
    load_and_prepare_data,
    save_model_artifacts,
)


warnings.filterwarnings(
    "ignore", message=".*Falling back to prediction using DMatrix.*"
)


def train_model(
    dataset_path="data/raw/dataset.csv",
    use_smote=True,
    smote_k_neighbors=5,
    use_chi2=True,
    chi2_k_features=20,
    use_gpu=True,
    output_dir="models",
):
    """Train model with default hyperparameters.

    Args:
        dataset_path: Path to the dataset CSV file
        use_smote: Whether to apply SMOTE for handling class imbalance (default: True)
        smote_k_neighbors: Number of nearest neighbors for SMOTE (default: 5)
        use_chi2: Whether to apply Chi-Squared feature selection (default: True)
        chi2_k_features: Number of top features to select with Chi-Squared (default: 20)
        use_gpu: Whether to use GPU acceleration (default: True)
        output_dir: Directory to save model artifacts (default: 'models')
    """

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        num_classes,
        le,
        scaler,
        selector,
    ) = load_and_prepare_data(
        dataset_path=dataset_path,
        use_smote=use_smote,
        smote_k_neighbors=smote_k_neighbors,
        use_chi2=use_chi2,
        chi2_k_features=chi2_k_features,
    )

    print("=" * 50)
    print("Training XGBoost Model")
    print("=" * 50)
    print(f"Using GPU acceleration: {'Yes' if use_gpu else 'No'}")

    params = get_base_params(num_classes, use_gpu=use_gpu)
    hyperparams = get_default_hyperparameters()
    params.update(hyperparams)

    print("\nModel parameters:")
    for key, value in sorted(params.items()):
        print(f"  {key}: {value}")
    print()

    if isinstance(X_train, pd.DataFrame):
        X_train_np = X_train.values
        X_val_np = X_val.values
        X_test_np = X_test.values
    else:
        X_train_np = np.asarray(X_train)
        X_val_np = np.asarray(X_val)
        X_test_np = np.asarray(X_test)

    model = xgb.XGBClassifier(**params)
    model.fit(X_train_np, y_train, eval_set=[(X_val_np, y_val)], verbose=True)

    print("\n" + "=" * 50)
    print("Validation Set Performance")
    print("=" * 50)
    y_val_pred = model.predict(X_val_np)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Validation Accuracy: {val_accuracy:.4f}\n")

    unique_labels = np.unique(np.concatenate([y_val, y_val_pred]))
    target_names_val = [
        str(le.classes_[i]).encode("ascii", "replace").decode("ascii")
        for i in unique_labels
    ]
    print(
        classification_report(
            y_val,
            y_val_pred,
            labels=unique_labels,
            target_names=target_names_val,
            zero_division=0,
        )
    )

    print("\n" + "=" * 50)
    print("Test Set Performance")
    print("=" * 50)
    y_test_pred = model.predict(X_test_np)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy: {test_accuracy:.4f}\n")

    unique_labels_test = np.unique(np.concatenate([y_test, y_test_pred]))
    target_names_test = [
        str(le.classes_[i]).encode("ascii", "replace").decode("ascii")
        for i in unique_labels_test
    ]
    print(
        classification_report(
            y_test,
            y_test_pred,
            labels=unique_labels_test,
            target_names=target_names_test,
            zero_division=0,
        )
    )

    # Save preprocessing configuration
    preprocessing_config = {
        "use_smote": use_smote,
        "smote_k_neighbors": smote_k_neighbors,
        "use_chi2": use_chi2,
        "chi2_k_features": chi2_k_features,
    }

    save_model_artifacts(
        model,
        le,
        hyperparameters=hyperparams,
        scaler=scaler,
        selector=selector,
        preprocessing_config=preprocessing_config,
        output_dir=output_dir,
    )


def main():
    """CLI entry point for model training."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train IDS machine learning model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  ids-train

  ids-train --no-gpu

  ids-train --chi2-features 30

  ids-train --no-chi2

  ids-train --no-smote

  ids-train --dataset data/raw/custom.csv --output-dir trained_models/
        """,
    )

    parser.add_argument(
        "--no-smote",
        action="store_true",
        help="Disable SMOTE oversampling for class imbalance",
    )

    parser.add_argument(
        "--smote-neighbors",
        type=int,
        default=5,
        help="Number of neighbors for SMOTE (default: 5)",
    )

    parser.add_argument(
        "--no-chi2", action="store_true", help="Disable Chi-Squared feature selection"
    )

    parser.add_argument(
        "--chi2-features",
        type=int,
        default=20,
        help="Number of top features to select with Chi-Squared (default: 20)",
    )

    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration (use CPU instead)",
    )

    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="data/raw/dataset.csv",
        help="Path to dataset CSV file (default: data/raw/dataset.csv)",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="models",
        help="Directory to save model artifacts (default: models)",
    )

    args = parser.parse_args()

    use_smote = not args.no_smote
    use_chi2 = not args.no_chi2
    use_gpu = not args.no_gpu

    print("\nTraining Configuration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  GPU acceleration: {'Enabled' if use_gpu else 'Disabled'}")
    print(f"  Chi-Squared feature selection: {'Enabled' if use_chi2 else 'Disabled'}")
    if use_chi2:
        print(f"  Chi-Squared k features: {args.chi2_features}")
    print(f"  SMOTE: {'Enabled' if use_smote else 'Disabled'}")
    if use_smote:
        print(f"  SMOTE neighbors: {args.smote_neighbors}")
    print()

    try:
        train_model(
            dataset_path=args.dataset,
            use_smote=use_smote,
            smote_k_neighbors=args.smote_neighbors,
            use_chi2=use_chi2,
            chi2_k_features=args.chi2_features,
            use_gpu=use_gpu,
            output_dir=args.output_dir,
        )
        return 0
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
