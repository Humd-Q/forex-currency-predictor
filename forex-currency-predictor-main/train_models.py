"""Train and save one forecasting artifact per FX currency.
Compares a persistence baseline with Ridge regression and saves the selected models and their evaluation metrics to the models directory. """
from pathlib import Path
import joblib
import pandas as pd
from forex_pipeline import currency_columns, evaluate_models, load_forex_data, train_artifact

MODELS_DIR = Path("models")

def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    data, all_metrics = load_forex_data(), []
    for currency in currency_columns(data):
        results, winner = evaluate_models(data[currency])
        all_metrics.append(results.assign(currency=currency))
        artifact = train_artifact(data[currency], winner)
        artifact["currency"] = currency
        joblib.dump(artifact, MODELS_DIR / f"{currency.replace('/', '_')}.joblib")
        print(f"Saved {currency}: {winner}")
    pd.concat(all_metrics, ignore_index=True).to_csv(MODELS_DIR / "model_metrics.csv", index=False)

if __name__ == "__main__":
    main()
