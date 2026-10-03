"""Streamlit interface for saved foreign-exchange forecasting models, must run train_models beforehand though."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from forex_pipeline import forecast_artifact, load_forex_data

MODELS_DIR = Path("models")
st.set_page_config(page_title="Forex Forecast", page_icon="📈", layout="wide")
st.title("Forex Currency Forecast")
st.caption("Saved, currency-specific models trained on the included historical data.")

@st.cache_data
def available_currencies() -> list[str]:
    return sorted(path.stem.replace("_", "/") for path in MODELS_DIR.glob("*.joblib"))

@st.cache_resource
def load_model(currency: str) -> dict:
    return joblib.load(MODELS_DIR / f"{currency.replace('/', '_')}.joblib")

currencies = available_currencies()
if not currencies:
    st.error("No saved models found. Run `python train_models.py` before starting the app.")
    st.stop()

currency = st.selectbox("Currency series", currencies)
horizon = st.slider("Forecast horizon (business days)", 1, 120, 30)
artifact = load_model(currency)
forecast = forecast_artifact(artifact, horizon)
historical = load_forex_data()[currency].tail(90).rename("historical").rename_axis("date").reset_index()

left, right = st.columns(2)
left.metric("Selected model", artifact["model_name"].replace("_", " ").title())
left.metric("Last observed date", pd.Timestamp(artifact["last_date"]).date().isoformat())
right.metric("Latest exchange rate", f"{artifact['history'][-1]:.4f}")
right.metric("Forecast end", f"{forecast.iloc[-1]['forecast']:.4f}")

chart = pd.concat([historical, forecast], ignore_index=True, sort=False).set_index("date")
st.line_chart(chart[["historical", "forecast"]])
st.subheader("Forecast values")
st.dataframe(forecast, hide_index=True, use_container_width=True)

with st.expander("Model evaluation"):
    metric_file = MODELS_DIR / "model_metrics.csv"
    if metric_file.exists():
        st.dataframe(pd.read_csv(metric_file).query("currency == @currency"), hide_index=True, use_container_width=True)
