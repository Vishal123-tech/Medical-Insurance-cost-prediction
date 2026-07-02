"""Central configuration: paths, feature definitions, and constants.

Keeping these in one place means every script (train, evaluate, predict, the app)
agrees on the same columns, split, and file locations.
"""
from __future__ import annotations

from pathlib import Path

# --- Paths -----------------------------------------------------------------
# PROJECT_ROOT is the folder that contains data/, models/, reports/, src/ ...
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

DATA_FILE = DATA_DIR / "insurance.csv"
MODEL_FILE = MODELS_DIR / "best_model.pkl"
METRICS_FILE = MODELS_DIR / "metrics.json"

# --- Features --------------------------------------------------------------
TARGET = "charges"
NUMERIC_FEATURES = ["age", "bmi", "children"]
CATEGORICAL_FEATURES = ["sex", "smoker", "region"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Valid categories (used by the app to build input widgets)
CATEGORIES = {
    "sex": ["female", "male"],
    "smoker": ["no", "yes"],
    "region": ["northeast", "northwest", "southeast", "southwest"],
}

# --- Modeling --------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# --- Display currency ------------------------------------------------------
# The dataset's `charges` are in US dollars. The app converts predictions to
# Indian Rupees for display only; the model is trained on the raw USD values.
CURRENCY_SYMBOL = "₹"          # ₹
USD_TO_INR = 83.0                    # display-only conversion rate


def ensure_dirs() -> None:
    """Create output directories if they don't already exist."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
