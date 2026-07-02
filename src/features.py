"""Preprocessing pipeline.

A single ColumnTransformer scales numeric columns and one-hot encodes the
categorical ones. It is wrapped inside every model Pipeline so that raw
(unencoded) records can be passed straight to `.predict()`.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def build_preprocessor() -> ColumnTransformer:
    """Return a ColumnTransformer for the numeric + categorical features."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), config.NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="if_binary"),
                config.CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
