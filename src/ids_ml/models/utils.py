"""Model training utilities."""

import json
import pickle
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from ids_ml.data.split import split_dataset


def load_and_prepare_data(
    dataset_path="data/raw/dataset.csv",
    use_smote=True,
    smote_k_neighbors=5,
    use_chi2=True,
    chi2_k_features=20,
):
    """Load data and prepare train/val/test splits.

    Args:
        dataset_path: Path to the dataset CSV file
        use_smote: Whether to apply SMOTE for handling class imbalance
        smote_k_neighbors: Number of nearest neighbors for SMOTE
        use_chi2: Whether to apply Chi-Squared feature selection (default: True)
        chi2_k_features: Number of top features to select with Chi-Squared (default: 20)

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test, num_classes, label_encoder, scaler, selector)
        where scaler and selector are the preprocessing objects (or None if not used)
    """
    df = pd.read_csv(dataset_path)
    num_classes = len(df["Label"].unique())

    le = LabelEncoder()

    train_df, test_df, val_df = split_dataset(csv_path=dataset_path)
    X_train = train_df.drop("Label", axis=1)
    y_train = le.fit_transform(train_df["Label"])

    X_val = val_df.drop("Label", axis=1)
    y_val = le.transform(val_df["Label"])

    X_test = test_df.drop("Label", axis=1)
    y_test = le.transform(test_df["Label"])

    scaler = None
    selector = None

    if use_chi2:
        print("\n" + "=" * 50)
        print("Applying Chi-Squared Feature Selection...")
        print("=" * 50)
        print(f"Original number of features: {X_train.shape[1]}")

        feature_names = X_train.columns.tolist()

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        k_features = min(chi2_k_features, X_train.shape[1])
        selector = SelectKBest(chi2, k=k_features)
        X_train = selector.fit_transform(X_train_scaled, y_train)
        X_val = selector.transform(X_val_scaled)
        X_test = selector.transform(X_test_scaled)

        selected_indices = selector.get_support(indices=True)
        selected_features = [feature_names[i] for i in selected_indices]

        print(f"Selected {k_features} features based on Chi-Squared scores")
        print("\nTop selected features:")

        scores = selector.scores_
        feature_scores = sorted(
            [(feature_names[i], scores[i]) for i in selected_indices],
            key=lambda x: x[1],
            reverse=True,
        )
        for feature, score in feature_scores[:10]:  # Show top 10
            print(f"  {feature}: {score:.2f}")
        if len(feature_scores) > 10:
            print(f"  ... and {len(feature_scores) - 10} more features")
        print("=" * 50 + "\n")

        X_train = pd.DataFrame(X_train, columns=selected_features)
        X_val = pd.DataFrame(X_val, columns=selected_features)
        X_test = pd.DataFrame(X_test, columns=selected_features)
    else:
        scaler = None
        selector = None

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
            class_name = (
                str(le.classes_[class_idx]).encode("ascii", "replace").decode("ascii")
            )
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
            class_name = (
                str(le.classes_[class_idx]).encode("ascii", "replace").decode("ascii")
            )
            print(
                f"  {class_name} (class {class_idx}): {count:,} samples ({count / len(y_train) * 100:.2f}%)"
            )
        print("=" * 50 + "\n")

    return (
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
    )


def get_base_params(num_classes, use_gpu=True):
    """Get base XGBoost parameters that are always used.

    Args:
        num_classes: Number of classes in the classification problem
        use_gpu: Whether to use GPU acceleration (default: True)

    Returns:
        Dictionary of base parameters
    """
    params = {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "random_state": 42,
        "early_stopping_rounds": 50,
        "tree_method": "hist",
    }

    if use_gpu:
        params["device"] = "cuda"
    else:
        params["device"] = "cpu"
        params["n_jobs"] = -1

    return params


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


def load_best_hyperparameters(filename="models/best_hyperparameters.json"):
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


def save_model_artifacts(
    model,
    label_encoder,
    hyperparameters=None,
    scaler=None,
    selector=None,
    preprocessing_config=None,
    output_dir="models",
):
    """Save model and related artifacts.

    Args:
        model: Trained XGBoost model
        label_encoder: Fitted LabelEncoder
        hyperparameters: Optional dict of hyperparameters to save
        scaler: Optional fitted MinMaxScaler
        selector: Optional fitted SelectKBest selector
        preprocessing_config: Dict with preprocessing settings (use_smote, use_chi2, etc.)
        output_dir: Directory to save artifacts (default: 'models')
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_model(str(output_dir / "model.json"))
    print(f"Model saved to {output_dir / 'model.json'}")

    label_mapping = {str(i): label for i, label in enumerate(label_encoder.classes_)}
    with open(output_dir / "label_mapping.json", "w") as f:
        json.dump(label_mapping, f, indent=2)
    print(f"Label mapping saved to {output_dir / 'label_mapping.json'}")

    if hyperparameters is not None:
        with open(output_dir / "best_hyperparameters.json", "w") as f:
            json.dump(hyperparameters, f, indent=2)
        print(
            f"Best hyperparameters saved to {output_dir / 'best_hyperparameters.json'}"
        )

    if preprocessing_config is not None:
        with open(output_dir / "preprocessing_config.json", "w") as f:
            json.dump(preprocessing_config, f, indent=2)
        print(
            f"Preprocessing configuration saved to {output_dir / 'preprocessing_config.json'}"
        )

    if scaler is not None:
        with open(output_dir / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved to {output_dir / 'scaler.pkl'}")

    if selector is not None:
        with open(output_dir / "selector.pkl", "wb") as f:
            pickle.dump(selector, f)
        print(f"Feature selector saved to {output_dir / 'selector.pkl'}")
