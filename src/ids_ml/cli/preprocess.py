"""Preprocess the CIC-IDS-2017 dataset for training."""

import sys
from pathlib import Path

from ids_ml.data.preprocess import merge_csv_files, preprocess_dataset


def main() -> None:
    """Merge and preprocess CSV files from the dataset."""
    # Define paths
    data_dir = Path("data")
    input_dir = data_dir / "MachineLearningCVE"
    raw_dir = data_dir / "raw"
    output_file = raw_dir / "dataset.csv"

    # Check if input directory exists
    if not input_dir.exists():
        print(f"ERROR: Input directory '{input_dir}' not found.")
        print("Please run 'uv run ids-download' first to download the dataset.")
        sys.exit(1)

    # Create raw directory if it doesn't exist
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CIC-IDS-2017 Dataset Preprocessing")
    print("=" * 60)
    print()

    # Step 1: Merge CSV files
    print("Step 1: Merging CSV files...")
    print("-" * 60)
    merge_csv_files(str(input_dir), str(output_file))
    print()

    # Step 2: Preprocess the merged dataset
    print("Step 2: Cleaning and preprocessing dataset...")
    print("-" * 60)
    preprocess_dataset(str(output_file))
    print()

    print("=" * 60)
    print("Preprocessing complete!")
    print(f"Dataset ready at: {output_file}")
    print()
    print("Next step: Run 'uv run ids-train' to train the model")
    print("=" * 60)


if __name__ == "__main__":
    main()
