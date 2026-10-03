# Forex Currency Predictor:

An end-to-end forecasting project for the 22 foreign-exchange series in `data/Foreign_Exchange_Rates.xls`. The CSV file's data is read accordingly.

## What it does:

- Cleans dates, removes export-only columns, converts invalid rate placeholders to missing values, and fills gaps using time interpolation.
- Evaluates three models independently for every currency on the final 60 observed days: persistence (last rate), lagged Ridge regression with weekday and annual-seasonality features, and ARIMA(1,1,1).
- Selects the lowest-RMSE model per currency, retrains it on all observations, and saves it in `models/`.
- Serves saved models through Streamlit; the app never retrains during a request, for speed and efficiency.

## Run locally:

```powershell
python -m pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

## Docker:

```powershell
docker build -t forex-predictor .
docker run --rm -p 8501:8501 forex-predictor
```

The Docker build trains the models so the container is ready to serve forecasts.

## Notebook:

`analysis.ipynb` documents preprocessing, EDA, model comparison, evaluation, and persistence. Run it from the project root after installing the requirements. Use Jupyter notebook or another environment such as Google Colab (ensure data is imported in).

By Humd Qazi
