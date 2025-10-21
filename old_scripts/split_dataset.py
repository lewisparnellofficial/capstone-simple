import pandas as pd
from sklearn.model_selection import train_test_split

from proportional_split import proportional_split


def split_dataset(
    proportions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    csv_path: str = ".\\dataset.csv",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path)

    split1, split2 = proportional_split(*proportions)

    train_df, temp_df = train_test_split(
        df, test_size=split1, random_state=random_state
    )
    test_df, val_df = train_test_split(
        temp_df, test_size=split2, random_state=random_state
    )

    return train_df, test_df, val_df
