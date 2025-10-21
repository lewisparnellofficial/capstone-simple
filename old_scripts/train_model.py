import optuna
import xgboost as xgb
from sklearn.metrics import accuracy_score

from train_utils import (
    get_base_params,
    load_and_prepare_data,
    save_model_artifacts,
)


def objective(trial, X_train, y_train, X_val, y_val, num_classes):
    """Optuna objective function for hyperparameter optimization."""

    # Hyperparameters to optimize
    hyperparams = {
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 150),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),
        "subsample": trial.suggest_float("subsample", 0.8, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.8, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 1),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 2),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 5),
    }

    # Combine with base parameters
    params = get_base_params(num_classes)
    params.update(hyperparams)

    model = xgb.XGBClassifier(**params)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    print(
        f"Trial {trial.number}: Accuracy = {accuracy:.4f}, Best iteration = {model.best_iteration}"
    )

    return accuracy


def train_model(
    n_trials=100, sample_fraction=None, use_smote=True, smote_k_neighbors=5
):
    """Train model with Optuna hyperparameter optimization.

    Args:
        n_trials: Number of Optuna trials to run
        sample_fraction: If provided (e.g., 0.2), use only this fraction of training data
                        for hyperparameter optimization. Final model uses all data.
        use_smote: Whether to apply SMOTE for handling class imbalance (default: True)
        smote_k_neighbors: Number of nearest neighbors for SMOTE (default: 5)
    """
    # Load and prepare data
    X_train, y_train, X_val, y_val, X_test, y_test, num_classes, le = (
        load_and_prepare_data(use_smote=use_smote, smote_k_neighbors=smote_k_neighbors)
    )

    if sample_fraction is not None and 0 < sample_fraction < 1:
        from sklearn.model_selection import train_test_split

        print(
            f"\nUsing {sample_fraction * 100:.0f}% of training data for hyperparameter optimization..."
        )
        print(f"Original training size: {len(X_train):,} samples")
        X_train_sample, _, y_train_sample, _ = train_test_split(
            X_train,
            y_train,
            train_size=sample_fraction,
            stratify=y_train,
            random_state=42,
        )
        print(f"Sampled training size: {len(X_train_sample):,} samples")
        X_train_opt, y_train_opt = X_train_sample, y_train_sample
    else:
        X_train_opt, y_train_opt = X_train, y_train

    print(f"Starting Optuna hyperparameter optimization with {n_trials} trials...")

    study = optuna.create_study(direction="maximize", study_name="xgboost_optimization")

    study.optimize(
        lambda trial: objective(
            trial, X_train_opt, y_train_opt, X_val, y_val, num_classes
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    print("\n" + "=" * 50)
    print("Optimization Complete!")
    print("=" * 50)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best validation accuracy: {study.best_value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    print("\nTraining final model with best hyperparameters on full training data...")

    # Combine best hyperparameters with base parameters
    params = get_base_params(num_classes)
    params.update(study.best_params)

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nFinal Test Accuracy: {accuracy:.4f}")

    # Save model and artifacts
    save_model_artifacts(model, le, hyperparameters=study.best_params)


if __name__ == "__main__":
    # Quick training with SMOTE for testing (20% data sample, 50 trials)
    train_model(n_trials=50, sample_fraction=0.2, use_smote=True, smote_k_neighbors=5)

    # Full training with SMOTE (uncomment to use all data and more trials)
    # train_model(n_trials=100, sample_fraction=None, use_smote=True, smote_k_neighbors=5)

    # Training without SMOTE (if you want to compare performance)
    # train_model(n_trials=50, sample_fraction=0.2, use_smote=False)
