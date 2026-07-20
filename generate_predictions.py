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


def generate_for_horizon(horizon: str) -> pd.DataFrame:
    latest = latest_feature_rows()
    tickers = latest["ticker"].tolist()
    # annualize daily volatility (~252 trading days) so risk reads in standard terms
    risk = (latest["volatility_21d"] * (252 ** 0.5)).tolist()
    
    # same feature prep as training
    feature_cols = select_available_features(latest, min_coverage=0.0)
    X, feature_cols = add_ticker_dummies(latest.copy(), feature_cols)
    X = X[feature_cols]

    model = load_model(horizon)

    # align columns to exactly what the model was trained on
    # (fills any missing dummy/feature columns with 0, drops extras)
    expected = getattr(model, "feature_names_in_", None)
    if expected is not None:
        X = X.reindex(columns=list(expected), fill_value=0)

    preds = model.predict(X)

    out = pd.DataFrame({
        "ticker": tickers,
        "predicted_return": preds,
        "predicted_risk": risk,
    })
    path = PROCESSED_DIR / f"final_predictions_{horizon}.parquet"
    out.to_parquet(path, index=False)
    print(f"[{horizon}] wrote {len(out)} rows to {path}")
    return out


if __name__ == "__main__":
    for horizon in HORIZONS:
        result = generate_for_horizon(horizon)
        print(result.sort_values("predicted_return", ascending=False).to_string(index=False))