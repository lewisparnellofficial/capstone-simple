import os

import numpy as np
import pandas as pd


def merge():
    csv_files = [
        f for f in os.listdir(".\\data\\MachineLearningCVE") if f.endswith(".csv")
    ]
    dfs = [pd.read_csv(f".\\data\\MachineLearningCVE\\{file}") for file in csv_files]

    df = pd.concat(dfs, ignore_index=True)
    df.to_csv("dataset.csv", index=False)


def preprocess():
    df = pd.read_csv("dataset.csv")

    df.columns = df.columns.str.strip()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    df.to_csv("dataset.csv", index=False)
