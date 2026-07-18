"""
Model 1: Regressor predicting expected forward return (%).

Trains on data/processed/training_set.parquet, produced by
build_features.py. Supports all three horizons from the worksheet
(1m/3m/6m) via the `horizon` argument -- these are different
prediction tasks, so a separate model is trained per horizon.

Feature set adapts to what's actually populated in the data right
now (see models/features.py) -- as fundamentals/news backfill over
time, re-running training automatically uses richer features with
no code changes needed.

XGBoost is the primary architecture; Random Forest is included as an
interpretable baseline for comparison in your writeup.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

sys.path.append(str(Path(__file__).parent.parent))
from config import PROCESSED_DIR
from storage import load_parquet
from models.features import HORIZONS, select_available_features


def load_training_data(horizon: str = "1m", min_coverage: float = 0.5):
    """
    Loads training_set.parquet, picks the feature columns that are
    actually populated (>= min_coverage), and returns (X, y, feature_cols)
    for the given horizon with incomplete rows dropped. Sorted
    chronologically -- required for time-series splits downstream.
    """
    if horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}, got {horizon!r}")

    df = load_parquet(PROCESSED_DIR / "training_set.parquet")
    if df.empty:
        raise FileNotFoundError(
            "data/processed/training_set.parquet not found or empty. "
            "Run build_features.py first (after collect_initial.py)."
        )

    df = df.sort_values("date")
    feature_cols = select_available_features(df, min_coverage=min_coverage)
    target_col = f"forward_return_{horizon}"

    df = df.dropna(subset=feature_cols + [target_col])
    return df[feature_cols], df[target_col], feature_cols


def train_xgb_regressor(X_train, y_train, X_val=None, y_val=None, params=None):
    default_params = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror",
        "random_state": 42,
    }
    if params:
        default_params.update(params)

    model = xgb.XGBRegressor(**default_params)
    eval_set = [(X_val, y_val)] if X_val is not None else None
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    return model


def train_rf_baseline(X_train, y_train):
    model = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def evaluate_regressor(model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "MAE": mean_absolute_error(y_test, preds),
        "R2": r2_score(y_test, preds),
    }


def time_series_cv_evaluate(X, y, model_fn=train_xgb_regressor, n_splits=5):
    """Time-aware CV -- never trains on future data to predict the past."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for train_idx, test_idx in tscv.split(X):
        model = model_fn(X.iloc[train_idx], y.iloc[train_idx])
        scores.append(evaluate_regressor(model, X.iloc[test_idx], y.iloc[test_idx]))
    return pd.DataFrame(scores)


def save_model(model, horizon: str = "1m", path: str | None = None):
    path = path or f"saved_models/regressor_{horizon}.joblib"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(horizon: str = "1m", path: str | None = None):
    path = path or f"saved_models/regressor_{horizon}.joblib"
    return joblib.load(path)