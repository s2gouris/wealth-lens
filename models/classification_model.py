"""
Model 2: Classifier predicting buy/hold/sell action (or growth probability).

Same data source and feature-adaptation approach as regression_model.py
-- see that file's docstring and models/features.py for details.

XGBoost is the primary architecture; Logistic Regression is included
as an interpretable baseline.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

sys.path.append(str(Path(__file__).parent.parent))
from config import PROCESSED_DIR
from storage import load_parquet
from models.features import HORIZONS, select_available_features, add_ticker_dummies
from models.engineered_features import add_engineered_features
from models.quantile_labels import add_quantile_labels


def load_training_data(horizon: str = "1m", min_coverage: float = 0.5,
                        use_ticker_dummies: bool = True, use_quantile_labels: bool = False):
    """
    Same feature pipeline as regression_model.load_training_data
    (engineered features + coverage-based selection + optional ticker
    dummies), but for the classification target.

    use_quantile_labels=True switches from the fixed +-5% threshold
    (label_{horizon}) to tertile cutoffs computed from the observed
    return distribution (label_{horizon}_quantile) -- see
    models/quantile_labels.py for why this addresses class imbalance.
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

    if use_quantile_labels:
        df, low_cut, high_cut = add_quantile_labels(df, horizon)
        target_col = f"label_{horizon}_quantile"
    else:
        target_col = f"label_{horizon}"

    feature_cols = select_available_features(df, min_coverage=min_coverage)
    df = df.dropna(subset=feature_cols + [target_col])
    if use_ticker_dummies:
        df, feature_cols = add_ticker_dummies(df, feature_cols)

    return df[feature_cols], df[target_col].astype(str), feature_cols


def train_xgb_classifier(X_train, y_train, params=None, use_class_weights=True):
    default_params = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "multi:softprob",
        "random_state": 42,
    }
    if params:
        default_params.update(params)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train)

    sample_weight = compute_sample_weight("balanced", y_encoded) if use_class_weights else None

    model = xgb.XGBClassifier(**default_params)
    model.fit(X_train, y_encoded, sample_weight=sample_weight)
    return model, le


def train_logistic_baseline(X_train, y_train):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y_encoded)
    return model, le, scaler


def evaluate_classifier(model, le, X_test, y_test):
    y_test_encoded = le.transform(y_test)
    preds = model.predict(X_test)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test_encoded, preds, average="weighted", zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_test_encoded, preds),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "report": classification_report(
            y_test_encoded, preds, target_names=le.classes_, zero_division=0
        ),
    }


def predict_growth_probability(model, le, X):
    """Probability of the 'buy' class -- useful for ranking stocks, not just labeling them."""
    probs = model.predict_proba(X)
    buy_idx = list(le.classes_).index("buy")
    return probs[:, buy_idx]


def time_series_cv_evaluate(X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []
    for train_idx, test_idx in tscv.split(X):
        model, le = train_xgb_classifier(X.iloc[train_idx], y.iloc[train_idx])
        metrics = evaluate_classifier(model, le, X.iloc[test_idx], y.iloc[test_idx])
        metrics.pop("report")
        results.append(metrics)
    return pd.DataFrame(results)


def save_model(model, le, horizon: str = "1m", path: str | None = None):
    """Bundles the model with its label encoder -- predictions are meaningless without it."""
    path = path or f"saved_models/classifier_{horizon}.joblib"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "label_encoder": le}, path)
    return path


def load_model(horizon: str = "1m", path: str | None = None):
    path = path or f"saved_models/classifier_{horizon}.joblib"
    bundle = joblib.load(path)
    return bundle["model"], bundle["label_encoder"]