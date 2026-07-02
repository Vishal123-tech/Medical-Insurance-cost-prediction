"""Score new records with the trained pipeline.

As a library:
    from src.predict import predict_one
    predict_one({"age": 30, "sex": "male", "bmi": 28.0,
                 "children": 1, "smoker": "no", "region": "northwest"})

As a script (runs a built-in smoke test comparing a smoker vs non-smoker):
    python -m src.predict
"""
from __future__ import annotations

import joblib
import pandas as pd

from . import config

_model = None


def _get_model():
    """Lazily load and cache the fitted pipeline."""
    global _model
    if _model is None:
        if not config.MODEL_FILE.exists():
            raise FileNotFoundError(
                f"{config.MODEL_FILE} not found. Run `python -m src.train` first."
            )
        _model = joblib.load(config.MODEL_FILE)
    return _model


def predict_one(record: dict) -> float:
    """Predict the annual charge for a single record (raw feature values)."""
    missing = [f for f in config.FEATURES if f not in record]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    row = pd.DataFrame([{f: record[f] for f in config.FEATURES}])
    return float(_get_model().predict(row)[0])


def predict_frame(df: pd.DataFrame) -> pd.Series:
    """Predict charges for every row of a DataFrame of raw features."""
    return pd.Series(_get_model().predict(df[config.FEATURES]), index=df.index)


def _smoke_test() -> None:
    base = {
        "age": 30,
        "sex": "male",
        "bmi": 28.0,
        "children": 1,
        "smoker": "no",
        "region": "northwest",
    }
    non_smoker = predict_one(base)
    smoker = predict_one({**base, "smoker": "yes"})
    print("Sample prediction (30yo male, bmi 28, 1 child, northwest):")
    print(f"  non-smoker: ${non_smoker:,.2f}")
    print(f"  smoker    : ${smoker:,.2f}")
    print(f"  difference: ${smoker - non_smoker:,.2f}")


if __name__ == "__main__":
    _smoke_test()
