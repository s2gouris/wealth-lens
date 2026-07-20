"""
Step 4: Generate the final predictions table for the UI.

Loads the trained regressors from saved_models/, runs each ticker's
most recent feature row through them, pairs the predicted return with
trailing volatility (volatility_21d) as the risk proxy, and saves
one predictions file per horizon:

    data/processed/final_predictions_{horizon}.parquet
    columns: ticker, predicted_return, predicted_risk

Run AFTER evaluate_all.py (models must exist in saved_models/).
"""
import pandas as pd

from config import PROCESSED_DIR
from storage import load_parquet
from models.features import HORIZONS, select_available_features, add_ticker_dummies
from models.engineered_features import add_engineered_features
from models.regression_model import load_model


def latest_feature_rows() -> pd.DataFrame:
    """One row per ticker: its most recent date in the feature table."""
    df = load_parquet(PROCESSED_DIR / "training_set.parquet")
    if df.empty:
        raise FileNotFoundError("training_set.parquet missing - run build_features.py first")

    df = add_engineered_features(df)
    df = df.sort_values("date")
    # last row per ticker = most recent known state of that stock
    return df.groupby("ticker").tail(1).reset_index(drop=True)


if __name__ == "__main__":
    latest = latest_feature_rows()
    print(latest[["ticker", "date", "volatility_21d"]].to_string(index=False))