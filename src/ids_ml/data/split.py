"""Dataset splitting utilities."""

import pandas as pd
from sklearn.model_selection import train_test_split


def proportional_split(
    proportion1: float, proportion2: float, proportion3: float
) -> tuple[float, float]:
    """Calculate split proportions for train_test_split.

    Args:
        proportion1: Proportion for first split (e.g., 0.8 for training)
        proportion2: Proportion for second split (e.g., 0.1 for test)
        proportion3: Proportion for third split (e.g., 0.1 for validation)

    Returns:
        Tuple of (split1, split2) proportions for train_test_split
    """
    split1 = proportion2 + proportion3
    split2 = proportion3 / split1
    return split1, split2


def split_dataset(
    proportions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    csv_path: str = "data/raw/dataset.csv",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into train, test, and validation sets.

    Args:
        proportions: Tuple of (train, test, val) proportions (default: 0.8, 0.1, 0.1)
        csv_path: Path to the dataset CSV file
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_df, test_df, val_df)
    """
    df = pd.read_csv(csv_path, low_memory=False)

    split1, split2 = proportional_split(*proportions)

    train_df, temp_df = train_test_split(
        df, test_size=split1, random_state=random_state
    )
    test_df, val_df = train_test_split(
        temp_df, test_size=split2, random_state=random_state
    )

    return train_df, test_df, val_df
