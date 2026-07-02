"""Evaluate the saved model on the held-out test set and write diagnostic plots.

Run with:  python -m src.evaluate
Requires that `python -m src.train` has been run first (needs models/best_model.pkl).
"""
from __future__ import annotations

import joblib
import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import config
from .data import load_data, split_data


def _load_model():
    if not config.MODEL_FILE.exists():
        raise FileNotFoundError(
            f"{config.MODEL_FILE} not found. Run `python -m src.train` first."
        )
    return joblib.load(config.MODEL_FILE)


def _plot_pred_vs_actual(y_true, y_pred, path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, edgecolor="none")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", lw=1.5, label="perfect")
    ax.set_xlabel("Actual charges")
    ax.set_ylabel("Predicted charges")
    ax.set_title("Predicted vs. Actual")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_residuals(y_true, y_pred, path):
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, residuals, alpha=0.5, edgecolor="none")
    ax.axhline(0, color="r", ls="--", lw=1.5)
    ax.set_xlabel("Predicted charges")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title("Residuals vs. Predicted")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_importance(model, X_test, y_test, path):
    result = permutation_importance(
        model, X_test, y_test, n_repeats=20, random_state=config.RANDOM_STATE, n_jobs=-1
    )
    order = result.importances_mean.argsort()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(
        np.array(config.FEATURES)[order],
        result.importances_mean[order],
        xerr=result.importances_std[order],
    )
    ax.set_xlabel("Mean drop in R² when shuffled")
    ax.set_title("Permutation feature importance")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    config.ensure_dirs()
    model = _load_model()
    df = load_data()
    _, X_test, _, y_test = split_data(df)

    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    print("Held-out test performance:")
    print(f"  R2  : {r2_score(y_test, y_pred):,.4f}")
    print(f"  RMSE: {rmse:,.4f}")
    print(f"  MAE : {mean_absolute_error(y_test, y_pred):,.4f}\n")

    _plot_pred_vs_actual(y_test, y_pred, config.FIGURES_DIR / "eval_pred_vs_actual.png")
    _plot_residuals(y_test, y_pred, config.FIGURES_DIR / "eval_residuals.png")
    _plot_importance(model, X_test, y_test, config.FIGURES_DIR / "eval_importance.png")
    print(f"Saved evaluation figures -> {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
