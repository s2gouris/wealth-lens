"""
Step 4: Generate the final predictions table for the UI.

Runs each ticker's most recent feature row through ALL THREE deployed
models and pairs the results into one decision-ready row per stock:

  - Regressor          -> predicted_return  (expected forward return %)
  - 3-class classifier -> signal            (buy / hold / sell)
                          prob_buy/hold/sell (class probabilities)
                          confidence         (max class probability)
  - Growth classifier  -> growth_prob       (P[forward return > 0])

Risk is trailing annualized volatility (volatility_21d), the same
risk proxy the allocator consumes.

Saves one predictions file per horizon:

    data/processed/final_predictions_{horizon}.parquet

Run AFTER evaluate_all.py (all three models must exist in saved_models/).
"""
import pandas as pd

from config import PROCESSED_DIR
from storage import load_parquet
from models.features import HORIZONS, select_available_features, add_ticker_dummies
from models.engineered_features import add_engineered_features
from models.regression_model import load_model as load_regressor
from models.classification_model import load_model as load_classifier
from models.growth_classifier import load_model as load_growth


def latest_feature_rows() -> pd.DataFrame:
    """One row per ticker: its most recent date in the feature table."""
    df = load_parquet(PROCESSED_DIR / "training_set.parquet")
    if df.empty:
        raise FileNotFoundError("training_set.parquet missing - run build_features.py first")

    df = add_engineered_features(df)
    df = df.sort_values("date")
    # last row per ticker = most recent known state of that stock
    return df.groupby("ticker").tail(1).reset_index(drop=True)


def _align_to_model(X: pd.DataFrame, model) -> pd.DataFrame:
    """
    Align columns to exactly what a model was trained on: fills any
    missing dummy/feature columns with 0 and drops extras, so a model
    trained on a different (past) feature snapshot still receives the
    columns it expects in the right order.
    """
    expected = getattr(model, "feature_names_in_", None)
    if expected is not None:
        return X.reindex(columns=list(expected), fill_value=0)
    return X


def generate_for_horizon(horizon: str) -> pd.DataFrame:
    latest = latest_feature_rows()
    tickers = latest["ticker"].tolist()
    # annualize daily volatility (~252 trading days) so risk reads in standard terms
    risk = (latest["volatility_21d"] * (252 ** 0.5)).tolist()

    # same feature prep as training: engineered features are already on
    # `latest`, now select populated columns and one-hot encode ticker
    feature_cols = select_available_features(latest, min_coverage=0.0)
    X_full, _ = add_ticker_dummies(latest.copy(), feature_cols)

    # --- Model 1: expected return (regression) ---
    regressor = load_regressor(horizon)
    returns = regressor.predict(_align_to_model(X_full, regressor))

    # --- Model 2: buy / hold / sell action (3-class classification) ---
    clf, le = load_classifier(horizon)
    X_clf = _align_to_model(X_full, clf)
    class_probs = clf.predict_proba(X_clf)
    signals = le.inverse_transform(clf.predict(X_clf))
    prob_by_class = {
        cls: class_probs[:, i] for i, cls in enumerate(le.classes_)
    }
    confidence = class_probs.max(axis=1)

    # --- Model 3: probability of positive growth (binary) ---
    growth_model = load_growth(horizon)
    growth_prob = growth_model.predict_proba(_align_to_model(X_full, growth_model))[:, 1]

    out = pd.DataFrame({
        "ticker": tickers,
        "predicted_return": returns,
        "predicted_risk": risk,
        "signal": signals,
        "confidence": confidence,
        "prob_buy": prob_by_class.get("buy"),
        "prob_hold": prob_by_class.get("hold"),
        "prob_sell": prob_by_class.get("sell"),
        "growth_prob": growth_prob,
    })
    path = PROCESSED_DIR / f"final_predictions_{horizon}.parquet"
    out.to_parquet(path, index=False)
    print(f"[{horizon}] wrote {len(out)} rows to {path}")
    return out


if __name__ == "__main__":
    for horizon in HORIZONS:
        result = generate_for_horizon(horizon)
        cols = ["ticker", "signal", "confidence", "growth_prob",
                "predicted_return", "predicted_risk"]
        print(result[cols].sort_values("growth_prob", ascending=False).to_string(index=False))
