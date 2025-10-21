"""Data preprocessing utilities."""

import os

import numpy as np
import pandas as pd


def merge_csv_files(
    input_dir="data/raw/MachineLearningCVE", output_file="data/raw/dataset.csv"
):
    """Merge multiple CSV files into a single dataset.

    Args:
        input_dir: Directory containing CSV files to merge
        output_file: Path to save the merged dataset
    """
    csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]
    print(f"Found {len(csv_files)} CSV files to merge")

    dfs = [pd.read_csv(os.path.join(input_dir, file)) for file in csv_files]

    df = pd.concat(dfs, ignore_index=True)
    print(f"Merged dataset has {len(df):,} rows")

    df.to_csv(output_file, index=False)
    print(f"Saved merged dataset to {output_file}")


def preprocess_dataset(input_file="data/raw/dataset.csv", output_file=None):
    """Preprocess dataset by cleaning column names and handling missing values.

    Args:
        input_file: Path to input dataset CSV
        output_file: Path to save preprocessed dataset (if None, overwrites input)
    """
    if output_file is None:
        output_file = input_file

    df = pd.read_csv(input_file, low_memory=False)
    print(f"Original dataset: {len(df):,} rows")

    df.columns = df.columns.str.strip()

    label_col = df["Label"]
    for col in df.columns:
        if col != "Label":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)

    rows_before = len(df)
    df = df.dropna()
    rows_after = len(df)
    print(f"Dropped {rows_before - rows_after:,} rows with NaN or malformed values")

    df.to_csv(output_file, index=False)
    print(f"Preprocessed dataset saved to {output_file} ({rows_after:,} rows)")
