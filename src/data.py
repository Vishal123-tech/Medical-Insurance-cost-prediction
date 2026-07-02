"""Data loading and train/test splitting."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def load_data(path=None) -> pd.DataFrame:
    """Load the raw insurance dataset as a DataFrame."""
    path = path or config.DATA_FILE
    df = pd.read_csv(path)
    # Drop exact duplicate rows (the dataset contains one known duplicate).
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def split_data(df: pd.DataFrame):
    """Split into X_train, X_test, y_train, y_test with a fixed seed.

    Returns raw (unencoded) feature frames — encoding happens inside the
    modeling pipeline so the same transformation is applied at predict time.
    """
    X = df[config.FEATURES].copy()
    y = df[config.TARGET].copy()
    return train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
