"""Train and compare models, then persist the best pipeline.

Run with:  python -m src.train

Steps:
  1. Load data, split into train/test.
  2. Cross-validate several models (all inside the shared preprocessing pipeline).
  3. Light hyper-parameter search on the best boosting model.
  4. Refit the winner on the full training set, evaluate on the held-out test set.
  5. Save the fitted pipeline (models/best_model.pkl) and metrics (models/metrics.json).
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline

from . import config
from .data import load_data, split_data
from .features import build_preprocessor


def _make_pipeline(model) -> Pipeline:
    """Wrap a regressor behind the shared preprocessor."""
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])


def _candidate_models() -> dict:
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, random_state=config.RANDOM_STATE, n_jobs=-1
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            random_state=config.RANDOM_STATE
        ),
    }


def compare_models(X_train, y_train) -> pd.DataFrame:
    """5-fold cross-validation of every candidate; return a summary table."""
    scoring = {"r2": "r2", "neg_rmse": "neg_root_mean_squared_error"}
    rows = []
    for name, model in _candidate_models().items():
        pipe = _make_pipeline(model)
        cv = cross_validate(
            pipe, X_train, y_train, cv=config.CV_FOLDS, scoring=scoring, n_jobs=-1
        )
        rows.append(
            {
                "model": name,
                "cv_r2_mean": cv["test_r2"].mean(),
                "cv_r2_std": cv["test_r2"].std(),
                "cv_rmse_mean": -cv["test_neg_rmse"].mean(),
                "cv_rmse_std": cv["test_neg_rmse"].std(),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("cv_r2_mean", ascending=False)
        .reset_index(drop=True)
    )


def tune_best(X_train, y_train) -> Pipeline:
    """Grid-search the HistGradientBoosting pipeline (the usual winner here)."""
    pipe = _make_pipeline(
        HistGradientBoostingRegressor(random_state=config.RANDOM_STATE)
    )
    param_grid = {
        "model__learning_rate": [0.05, 0.1, 0.2],
        "model__max_iter": [200, 400],
        "model__max_leaf_nodes": [15, 31],
        "model__l2_regularization": [0.0, 1.0],
    }
    search = GridSearchCV(
        pipe,
        param_grid,
        scoring="r2",
        cv=config.CV_FOLDS,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search


def _metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def main() -> None:
    config.ensure_dirs()
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    print(f"Loaded {len(df)} rows  |  train={len(X_train)}  test={len(X_test)}\n")

    print("Cross-validated model comparison (5-fold, on training set):")
    comparison = compare_models(X_train, y_train)
    with pd.option_context("display.float_format", lambda v: f"{v:,.3f}"):
        print(comparison.to_string(index=False))
    print()

    print("Tuning HistGradientBoosting via GridSearchCV ...")
    search = tune_best(X_train, y_train)
    best_pipe = search.best_estimator_
    print(f"Best params: {search.best_params_}")
    print(f"Best CV R^2: {search.best_score_:.4f}\n")

    # Held-out test performance.
    test_pred = best_pipe.predict(X_test)
    test_metrics = _metrics(y_test, test_pred)
    print("Held-out test performance:")
    for k, v in test_metrics.items():
        print(f"  {k.upper():5s}: {v:,.4f}")

    # Persist artifacts.
    joblib.dump(best_pipe, config.MODEL_FILE)
    payload = {
        "cv_comparison": comparison.to_dict(orient="records"),
        "best_params": search.best_params_,
        "best_cv_r2": float(search.best_score_),
        "test_metrics": test_metrics,
        "n_rows": int(len(df)),
    }
    with open(config.METRICS_FILE, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"\nSaved model  -> {config.MODEL_FILE}")
    print(f"Saved metrics-> {config.METRICS_FILE}")


if __name__ == "__main__":
    main()
