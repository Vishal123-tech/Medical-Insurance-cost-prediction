"""Interactive insurance cost predictor (Streamlit).

Run with:  streamlit run app.py
Requires models/best_model.pkl (create it with `python -m src.train`).
"""
from __future__ import annotations

import json

import joblib
import pandas as pd
import streamlit as st

from src import config

st.set_page_config(page_title="Insurance Cost Predictor", page_icon="💸", layout="centered")


def to_inr(usd: float) -> float:
    """Convert a USD amount to Indian Rupees for display."""
    return usd * config.USD_TO_INR


def fmt_inr(usd: float) -> str:
    """Format a USD amount as Rupees with Indian-style digit grouping.

    e.g. 18882.57 USD -> '₹15,67,254' (whole rupees).
    """
    rupees = int(round(to_inr(usd)))
    sign = "-" if rupees < 0 else ""
    digits = str(abs(rupees))
    if len(digits) > 3:
        last3 = digits[-3:]
        rest = digits[:-3]
        # group the remaining digits in pairs (Indian lakh/crore system)
        parts = []
        while len(rest) > 2:
            parts.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.insert(0, rest)
        grouped = ",".join(parts) + "," + last3
    else:
        grouped = digits
    return f"{sign}{config.CURRENCY_SYMBOL}{grouped}"


@st.cache_resource
def load_model():
    if not config.MODEL_FILE.exists():
        return None
    return joblib.load(config.MODEL_FILE)


@st.cache_data
def load_metrics():
    if config.METRICS_FILE.exists():
        return json.loads(config.METRICS_FILE.read_text())
    return None


st.title("💸 Medical Insurance Cost Predictor")
st.write(
    "Estimate an individual's **annual medical insurance charge** from their "
    "demographics and lifestyle. Model: HistGradientBoosting trained on the "
    "Kaggle *Medical Cost Personal* dataset."
)

model = load_model()
if model is None:
    st.error("No trained model found. Run `python -m src.train` first, then reload.")
    st.stop()

# --- Inputs ----------------------------------------------------------------
st.sidebar.header("Enter details")
age = st.sidebar.slider("Age", 18, 64, 30)
sex = st.sidebar.selectbox("Sex", config.CATEGORIES["sex"])
bmi = st.sidebar.slider("BMI", 15.0, 55.0, 28.0, step=0.1)
children = st.sidebar.slider("Number of children", 0, 5, 0)
smoker = st.sidebar.selectbox("Smoker", config.CATEGORIES["smoker"])
region = st.sidebar.selectbox("Region", config.CATEGORIES["region"])

record = {
    "age": age,
    "sex": sex,
    "bmi": bmi,
    "children": children,
    "smoker": smoker,
    "region": region,
}
X = pd.DataFrame([record])[config.FEATURES]
prediction = float(model.predict(X)[0])

# --- Output ----------------------------------------------------------------
st.subheader("Predicted annual charge")
st.metric(label="Estimated cost", value=fmt_inr(prediction))

# Show the smoker effect: same person, flipped smoker status.
counterfactual = {**record, "smoker": "no" if smoker == "yes" else "yes"}
cf_pred = float(model.predict(pd.DataFrame([counterfactual])[config.FEATURES])[0])
delta = prediction - cf_pred
st.caption(
    f"If this person were a {'non-smoker' if smoker == 'yes' else 'smoker'}, the "
    f"estimate would be **{fmt_inr(cf_pred)}** "
    f"({'−' if delta > 0 else '+'}{fmt_inr(abs(delta))})."
)

with st.expander("Your inputs"):
    st.table(pd.DataFrame([record]))

metrics = load_metrics()
if metrics:
    tm = metrics.get("test_metrics", {})
    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"**Model quality (held-out test)**\n\n"
        f"- R² = {tm.get('r2', float('nan')):.3f}\n"
        f"- RMSE = {fmt_inr(tm.get('rmse', float('nan')))}\n"
        f"- MAE = {fmt_inr(tm.get('mae', float('nan')))}"
    )
    st.sidebar.caption(f"Converted at ₹{config.USD_TO_INR:,.0f} / USD (display only).")
