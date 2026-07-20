"""
Model 2b: Binary growth classifier -- predicts whether a stock's
forward return will be positive ("growth") or not, rather than the
3-class buy/hold/sell split.

This directly matches the worksheet's own Prediction Task section,
which lists "Probability of stock growth" as a valid Y variable
alongside buy/hold/sell -- it's not a reframing invented to inflate
a number, it's one of the two output types the worksheet specifies.

Why this is a legitimate, different task rather than an easier
version of the same one: 3-class buy/hold/sell asks the model to
also separate "meaningfully up" from "flat" from "meaningfully down"
-- three decision boundaries. Binary growth only asks "up or not,"
one boundary. Baseline (random guessing) is 50% here vs ~33% for
3-class, and accuracy numbers from the two tasks are not directly
comparable -- report which task a number came from.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

sys.path.append(str(Path(__file__).parent.parent))
from config import PROCESSED_DIR
from storage import load_parquet
from models.features import HORIZONS, select_available_features, add_ticker_dummies
from models.engineered_features import add_engineered_features


def load_growth_data(horizon: str = "1m", min_coverage: float = 0.5, use_ticker_dummies: bool = True):
    """
    Same feature pipeline as classification_model.load_training_data,
    but the target is binary: 1 if forward_return_{horizon} > 0
    ("growth"), 0 otherwise.
    """
    if horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}, got {horizon!r}")

    df = load_parquet(PROCESSED_DIR / "training_set.parquet")
    if df.empty:
        raise FileNotFoundError(
            "data/processed/training_set.parquet not found or empty. "
            "Run build_features.py first (after collect_initial.py)."
        )

    df = add_engineered_features(df)
    df = df.sort_values("date")

    return_col = f"forward_return_{horizon}"
    feature_cols = select_available_features(df, min_coverage=min_coverage)
    df = df.dropna(subset=feature_cols + [return_col])
    df["growth_label"] = (df[return_col] > 0).astype(int)

    if use_ticker_dummies:
        df, feature_cols = add_ticker_dummies(df, feature_cols)

    return df[feature_cols], df["growth_label"], feature_cols


def predict_growth_probability(model, X):
    """
    Returns the model's predicted probability of growth (class 1)
    for each row -- a continuous score in [0, 1], useful for ranking
    stocks by confidence rather than just a hard growth/no-growth
    label. This is the worksheet's "Probability of stock growth" Y
    variable directly.
    """
    return model.predict_proba(X)[:, 1]


def train_growth_classifier(X_train, y_train, params=None, use_class_weights=True):
    default_params = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "random_state": 42,
    }
    if params:
        default_params.update(params)

    sample_weight = compute_sample_weight("balanced", y_train) if use_class_weights else None

    model = xgb.XGBClassifier(**default_params)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def evaluate_growth_classifier(model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, preds, average="binary", zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc_score(y_test, probs),
        "report": classification_report(y_test, preds, target_names=["no_growth", "growth"], zero_division=0),
    }


def time_series_cv_evaluate(X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []
    for train_idx, test_idx in tscv.split(X):
        model = train_growth_classifier(X.iloc[train_idx], y.iloc[train_idx])
        metrics = evaluate_growth_classifier(model, X.iloc[test_idx], y.iloc[test_idx])
        metrics.pop("report")
        results.append(metrics)
    return pd.DataFrame(results)


def save_model(model, horizon: str = "1m", path: str | None = None):
    path = path or f"saved_models/growth_classifier_{horizon}.joblib"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(horizon: str = "1m", path: str | None = None):
    path = path or f"saved_models/growth_classifier_{horizon}.joblib"
    return joblib.load(path)