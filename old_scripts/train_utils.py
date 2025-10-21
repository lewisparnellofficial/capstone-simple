"""Shared utilities for training models."""

import json

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

from split_dataset import split_dataset


def load_and_prepare_data(use_smote=True, smote_k_neighbors=5):
    """Load data and prepare train/val/test splits.

    Args:
        use_smote: Whether to apply SMOTE for handling class imbalance
        smote_k_neighbors: Number of nearest neighbors for SMOTE

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test, num_classes, label_encoder)
    """
    df = pd.read_csv(".\\dataset.csv")
    num_classes = len(df["Label"].unique())

    le = LabelEncoder()

    train_df, test_df, val_df = split_dataset()
    X_train = train_df.drop("Label", axis=1)
    y_train = le.fit_transform(train_df["Label"])

    X_val = val_df.drop("Label", axis=1)
    y_val = le.transform(val_df["Label"])

    X_test = test_df.drop("Label", axis=1)
    y_test = le.transform(test_df["Label"])

    if use_smote:
        print("\n" + "=" * 50)
        print("Applying SMOTE to balance training data...")
        print("=" * 50)
        print(f"Original training size: {len(X_train):,} samples")
        print("Class distribution before SMOTE:")
        unique, counts = (
            pd.Series(y_train).value_counts().sort_index().index,
            pd.Series(y_train).value_counts().sort_index().values,
        )
        for class_idx, count in zip(unique, counts):
            class_name = le.classes_[class_idx]
            print(
                f"  {class_name} (class {class_idx}): {count:,} samples ({count / len(y_train) * 100:.2f}%)"
            )

        smote = SMOTE(random_state=42, k_neighbors=smote_k_neighbors)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        print(f"\nAfter SMOTE training size: {len(X_train):,} samples")
        print("Class distribution after SMOTE:")
        unique, counts = (
            pd.Series(y_train).value_counts().sort_index().index,
            pd.Series(y_train).value_counts().sort_index().values,
        )
        for class_idx, count in zip(unique, counts):
            class_name = le.classes_[class_idx]
            print(
                f"  {class_name} (class {class_idx}): {count:,} samples ({count / len(y_train) * 100:.2f}%)"
            )
        print("=" * 50 + "\n")

    return X_train, y_train, X_val, y_val, X_test, y_test, num_classes, le


def get_base_params(num_classes):
    """Get base XGBoost parameters that are always used.

    Args:
        num_classes: Number of classes in the classification problem

    Returns:
        Dictionary of base parameters
    """
    return {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "tree_method": "hist",
        "random_state": 42,
        "early_stopping_rounds": 50,
        "n_jobs": -1,
    }


def get_default_hyperparameters():
    """Get default hyperparameters for XGBoost.

    Returns:
        Dictionary of default hyperparameters
    """
    return {
        "max_depth": 5,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0.1,
        "reg_alpha": 0.5,
        "reg_lambda": 2,
    }


def load_best_hyperparameters(filename="best_hyperparameters.json"):
    """Load best hyperparameters from file, or return defaults if not found.

    Args:
        filename: Path to the hyperparameters JSON file

    Returns:
        Dictionary of hyperparameters
    """
    try:
        with open(filename, "r") as f:
            params = json.load(f)
        print(f"Loaded hyperparameters from {filename}")
        return params
    except FileNotFoundError:
        print(f"{filename} not found, using default hyperparameters")
        return get_default_hyperparameters()


def save_model_artifacts(model, label_encoder, hyperparameters=None):
    """Save model and related artifacts.

    Args:
        model: Trained XGBoost model
        label_encoder: Fitted LabelEncoder
        hyperparameters: Optional dict of hyperparameters to save
    """
    model.save_model("model.json")
    print("Model saved to model.json")

    label_mapping = {str(i): label for i, label in enumerate(label_encoder.classes_)}
    with open("label_mapping.json", "w") as f:
        json.dump(label_mapping, f, indent=2)
    print("Label mapping saved to label_mapping.json")

    if hyperparameters is not None:
        with open("best_hyperparameters.json", "w") as f:
            json.dump(hyperparameters, f, indent=2)
        print("Best hyperparameters saved to best_hyperparameters.json")
