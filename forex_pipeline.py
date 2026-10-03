"""Reusable data preparation, evaluation, and forecasting helpers."""
from __future__ import annotations
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from statsmodels.tsa.arima.model import ARIMA

DATA_PATH = Path("data/Foreign_Exchange_Rates.xls")
DATE_COLUMN, TEST_DAYS, LAGS = "Time Serie", 60, (1, 5, 20)

def load_forex_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load the supplied CSV-formatted .xls file and clean missing values."""
    frame = pd.read_csv(path)
    frame = frame.loc[:, ~frame.columns.str.contains(r"^Unnamed")].copy()
    frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN], format="%d-%m-%Y", errors="coerce")
    frame = frame.dropna(subset=[DATE_COLUMN]).sort_values(DATE_COLUMN)
    frame = frame.drop_duplicates(subset=[DATE_COLUMN], keep="last").set_index(DATE_COLUMN)
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.interpolate(method="time", limit_direction="both").ffill().bfill()

def currency_columns(frame: pd.DataFrame) -> list[str]:
    return frame.select_dtypes(include="number").columns.tolist()

def make_features(history: Iterable[float], date: pd.Timestamp) -> np.ndarray:
    values = np.asarray(list(history), dtype=float)
    if len(values) < max(LAGS):
        raise ValueError("At least 20 observations are required.")
    doy = date.dayofyear
    return np.array([*(values[-lag] for lag in LAGS), np.sin(2*np.pi*date.dayofweek/7), np.cos(2*np.pi*date.dayofweek/7), np.sin(2*np.pi*doy/365.25), np.cos(2*np.pi*doy/365.25)]).reshape(1, -1)

def fit_ridge(series: pd.Series, alpha: float = 1.0) -> Ridge:
    values, dates = series.to_numpy(float), series.index
    rows, target = [], []
    for index in range(max(LAGS), len(values)):
        rows.append(make_features(values[:index], dates[index]).ravel())
        target.append(values[index])
    return Ridge(alpha=alpha).fit(np.asarray(rows), np.asarray(target))

def fit_arima(series: pd.Series):
    """Fit a conventional non-seasonal ARIMA(1,1,1) time-series model."""
    return ARIMA(series, order=(1, 1, 1), trend="t").fit()

def recursive_forecast(model: Ridge, history: Iterable[float], start_date: pd.Timestamp, horizon: int) -> pd.Series:
    values, predictions = list(np.asarray(list(history), float)), []
    dates = pd.bdate_range(start_date + pd.offsets.BDay(1), periods=horizon)
    for date in dates:
        value = float(model.predict(make_features(values, date))[0])
        predictions.append(value); values.append(value)
    return pd.Series(predictions, index=dates, name="forecast")

def metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    observed, estimates = actual.to_numpy(float), predicted.to_numpy(float)
    errors = observed - estimates
    nonzero = np.abs(observed) > 1e-12
    return {"mae": float(np.mean(np.abs(errors))), "rmse": float(np.sqrt(np.mean(errors**2))), "mape": float(np.mean(np.abs(errors[nonzero] / observed[nonzero])) * 100)}

def evaluate_models(series: pd.Series, test_days: int = TEST_DAYS) -> tuple[pd.DataFrame, str]:
    train, test = series.iloc[:-test_days], series.iloc[-test_days:]
    naive = pd.Series(train.iloc[-1], index=test.index)
    prediction = recursive_forecast(fit_ridge(train), train, train.index[-1], len(test))
    prediction.index = test.index
    arima_prediction = pd.Series(fit_arima(train).forecast(steps=len(test)).to_numpy(), index=test.index)
    candidates = {"persistence": naive, "ridge_lagged": prediction, "arima_1_1_1": arima_prediction}
    results = pd.DataFrame([{"model": name, **metrics(test, output)} for name, output in candidates.items()]).sort_values("rmse").reset_index(drop=True)
    return results, str(results.iloc[0]["model"])

def train_artifact(series: pd.Series, model_name: str) -> dict:
    if model_name == "persistence":
        model = None
    elif model_name == "ridge_lagged":
        model = fit_ridge(series)
    elif model_name == "arima_1_1_1":
        model = fit_arima(series)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return {"model_name": model_name, "model": model, "history": series.iloc[-max(LAGS):].to_list(), "last_date": series.index[-1]}

def forecast_artifact(artifact: dict, horizon: int) -> pd.DataFrame:
    last_date = pd.Timestamp(artifact["last_date"])
    dates = pd.bdate_range(last_date + pd.offsets.BDay(1), periods=horizon)
    if artifact["model_name"] == "persistence":
        forecast = pd.Series(artifact["history"][-1], index=dates, name="forecast")
    elif artifact["model_name"] == "arima_1_1_1":
        forecast = pd.Series(artifact["model"].forecast(steps=horizon).to_numpy(), index=dates, name="forecast")
    else:
        forecast = recursive_forecast(artifact["model"], artifact["history"], last_date, horizon)
    return forecast.rename_axis("date").reset_index()
