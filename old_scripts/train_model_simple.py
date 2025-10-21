"""Simple training script without Optuna hyperparameter optimization.

This script trains an XGBoost model using either:
1. Previously optimized hyperparameters from best_hyperparameters.json
2. Sensible default hyperparameters

Use this for faster training when you already have good hyperparameters or want to quickly test changes.
"""

import xgboost as xgb
from sklearn.metrics import accuracy_score

from train_utils import (
    get_base_params,
    load_and_prepare_data,
    load_best_hyperparameters,
    save_model_artifacts,
)


def train_model_simple(use_smote=True, smote_k_neighbors=5, hyperparameters=None):
    """Train model with fixed hyperparameters (no Optuna optimization).

    Args:
        use_smote: Whether to apply SMOTE for handling class imbalance (default: True)
        smote_k_neighbors: Number of nearest neighbors for SMOTE (default: 5)
        hyperparameters: Optional dict of hyperparameters to use. If None, loads from
                        best_hyperparameters.json or uses defaults.

    Returns:
        Trained XGBoost model
    """
    # Load and prepare data
    X_train, y_train, X_val, y_val, X_test, y_test, num_classes, le = (
        load_and_prepare_data(use_smote=use_smote, smote_k_neighbors=smote_k_neighbors)
    )

    # Get hyperparameters
    if hyperparameters is None:
        hyperparameters = load_best_hyperparameters()
    else:
        print("Using provided hyperparameters")

    # Combine with base parameters
    params = get_base_params(num_classes)
    params.update(hyperparameters)

    print("\n" + "=" * 50)
    print("Training XGBoost model with hyperparameters:")
    print("=" * 50)
    for key, value in hyperparameters.items():
        print(f"  {key}: {value}")
    print("=" * 50 + "\n")

    # Train model
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 50)
    print(f"Test Accuracy: {accuracy:.4f}")
    print("=" * 50 + "\n")

    # Save artifacts
    save_model_artifacts(model, le)

    return model


if __name__ == "__main__":
    # Train with SMOTE using best/default hyperparameters
    train_model_simple(use_smote=True, smote_k_neighbors=5)

    # Alternative: Train without SMOTE
    # train_model_simple(use_smote=False)

    # Alternative: Train with custom hyperparameters
    # custom_params = {
    #     "max_depth": 6,
    #     "learning_rate": 0.05,
    #     "n_estimators": 150,
    #     "min_child_weight": 2,
    #     "subsample": 0.9,
    #     "colsample_bytree": 0.9,
    #     "gamma": 0.2,
    #     "reg_alpha": 1.0,
    #     "reg_lambda": 3.0,
    # }
    # train_model_simple(use_smote=True, hyperparameters=custom_params)
