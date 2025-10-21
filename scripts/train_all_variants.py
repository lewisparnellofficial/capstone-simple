"""Train all model variants with different preprocessing configurations."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ids_ml.models.train import train_model


def main():
    """Train all combinations of SMOTE and Chi2 feature selection."""

    configurations = [
        {
            "name": "smote_chi2",
            "use_smote": True,
            "use_chi2": True,
            "output_dir": "models/smote_chi2"
        },
        {
            "name": "smote_no_chi2",
            "use_smote": True,
            "use_chi2": False,
            "output_dir": "models/smote_no_chi2"
        },
        {
            "name": "no_smote_chi2",
            "use_smote": False,
            "use_chi2": True,
            "output_dir": "models/no_smote_chi2"
        },
        {
            "name": "no_smote_no_chi2",
            "use_smote": False,
            "use_chi2": False,
            "output_dir": "models/no_smote_no_chi2"
        }
    ]

    dataset_path = "data/raw/dataset.csv"

    print("=" * 70)
    print("TRAINING ALL MODEL VARIANTS")
    print("=" * 70)
    print(f"\nDataset: {dataset_path}")
    print(f"Number of configurations: {len(configurations)}\n")

    for i, config in enumerate(configurations, 1):
        print("\n" + "=" * 70)
        print(f"Configuration {i}/{len(configurations)}: {config['name']}")
        print("=" * 70)
        print(f"  SMOTE: {'Enabled' if config['use_smote'] else 'Disabled'}")
        print(f"  Chi-Squared: {'Enabled' if config['use_chi2'] else 'Disabled'}")
        print(f"  Output directory: {config['output_dir']}")
        print()

        try:
            train_model(
                dataset_path=dataset_path,
                use_smote=config["use_smote"],
                smote_k_neighbors=5,
                use_chi2=config["use_chi2"],
                chi2_k_features=20,
                use_gpu=True,
                output_dir=config["output_dir"]
            )
            print(f"\n✓ Configuration '{config['name']}' completed successfully!")
        except Exception as e:
            print(f"\n✗ Configuration '{config['name']}' failed with error: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 70)
    print("ALL TRAINING COMPLETED")
    print("=" * 70)
    print("\nYou can now run the comparison script to evaluate all variants:")
    print("  python scripts/compare_variants.py")


if __name__ == "__main__":
    main()
