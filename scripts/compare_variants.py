"""Compare performance of all model variants."""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ids_ml.models.evaluate import evaluate_model


def load_preprocessing_config(model_dir):
    """Load preprocessing configuration for a model."""
    config_path = Path(model_dir) / "preprocessing_config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)
    return None


def format_config_name(config):
    """Format configuration into a readable name."""
    if config is None:
        return "Unknown Configuration"

    smote = "SMOTE" if config.get("use_smote", False) else "No SMOTE"
    chi2 = "Chi2" if config.get("use_chi2", False) else "No Chi2"
    return f"{smote} + {chi2}"


def main():
    """Evaluate and compare all model variants."""

    # Define model directories to compare
    model_dirs = [
        "models/smote_chi2",
        "models/smote_no_chi2",
        "models/no_smote_chi2",
        "models/no_smote_no_chi2"
    ]

    # Also check the default models directory
    if Path("models/model.json").exists():
        model_dirs.insert(0, "models")

    dataset_path = "data/raw/dataset.csv"
    results = []

    print("=" * 80)
    print("MODEL VARIANT COMPARISON")
    print("=" * 80)
    print(f"\nDataset: {dataset_path}\n")

    for model_dir in model_dirs:
        model_path = Path(model_dir)

        # Check if model exists
        if not (model_path / "model.json").exists():
            print(f"Skipping {model_dir} - model not found")
            continue

        # Load preprocessing config
        config = load_preprocessing_config(model_dir)
        config_name = format_config_name(config)

        print("\n" + "=" * 80)
        print(f"Evaluating: {config_name}")
        print(f"Model directory: {model_dir}")
        print("=" * 80)

        try:
            # Evaluate the model
            metrics = evaluate_model(
                dataset_path=dataset_path,
                model_dir=str(model_dir),
                output_file=str(model_path / "model_metrics.json")
            )

            # Store results
            results.append({
                "config_name": config_name,
                "model_dir": model_dir,
                "config": config,
                "metrics": metrics
            })

        except Exception as e:
            print(f"\nError evaluating {model_dir}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Display comparison table
    if results:
        print("\n\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)
        print()
        print(f"{'Configuration':<30} {'Accuracy':<12} {'Weighted F1':<12} {'Macro F1':<12}")
        print("-" * 80)

        # Sort by accuracy (descending)
        results.sort(key=lambda x: x["metrics"]["accuracy"], reverse=True)

        for result in results:
            config_name = result["config_name"]
            accuracy = result["metrics"]["accuracy"]
            weighted_f1 = result["metrics"]["weighted_metrics"]["f1_score"]
            macro_f1 = result["metrics"]["macro_metrics"]["f1_score"]

            print(f"{config_name:<30} {accuracy:<12.4f} {weighted_f1:<12.4f} {macro_f1:<12.4f}")

        print()

        # Identify best models
        best_accuracy = max(results, key=lambda x: x["metrics"]["accuracy"])
        best_weighted_f1 = max(results, key=lambda x: x["metrics"]["weighted_metrics"]["f1_score"])
        best_macro_f1 = max(results, key=lambda x: x["metrics"]["macro_metrics"]["f1_score"])

        print("=" * 80)
        print("BEST PERFORMING MODELS")
        print("=" * 80)
        print(f"Best Accuracy:    {best_accuracy['config_name']} ({best_accuracy['metrics']['accuracy']:.4f})")
        print(f"Best Weighted F1: {best_weighted_f1['config_name']} ({best_weighted_f1['metrics']['weighted_metrics']['f1_score']:.4f})")
        print(f"Best Macro F1:    {best_macro_f1['config_name']} ({best_macro_f1['metrics']['macro_metrics']['f1_score']:.4f})")
        print()

        # Save comparison results
        comparison_file = "models/comparison_results.json"
        with open(comparison_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Detailed comparison results saved to {comparison_file}")

    else:
        print("\nNo models found to compare. Please train models first using:")
        print("  python scripts/train_all_variants.py")


if __name__ == "__main__":
    main()
