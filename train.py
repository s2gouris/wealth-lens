"""
Trains the regressor and classifier for all three worksheet horizons
(1m/3m/6m) on data/processed/training_set.parquet, and saves each to
saved_models/. Run build_features.py first.

Feature columns are chosen automatically based on what's actually
populated in the data right now (see models/features.py) -- as
fundamentals and news sentiment backfill over time, just re-run this
script and it picks up the richer feature set with no code changes.
"""
from models.regression_model import (
    load_training_data as load_regression_data,
    train_xgb_regressor,
    evaluate_regressor,
    save_model as save_regressor,
)
from models.classification_model import (
    load_training_data as load_classification_data,
    train_xgb_classifier,
    evaluate_classifier,
    save_model as save_classifier,
)
from models.features import HORIZONS


def train_horizon(horizon: str):
    print(f"\n{'=' * 50}")
    print(f"Horizon: {horizon}")
    print('=' * 50)

    # --- Regressor ---
    X, y, feature_cols = load_regression_data(horizon)
    print(f"Regressor features used ({len(feature_cols)}): {feature_cols}")
    print(f"Rows after dropping incomplete: {len(X)}")

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    reg_model = train_xgb_regressor(X_train, y_train)
    reg_metrics = evaluate_regressor(reg_model, X_test, y_test)
    print(f"Regression metrics: {reg_metrics}")
    reg_path = save_regressor(reg_model, horizon)
    print(f"Saved to: {reg_path}")

    # --- Classifier ---
    Xc, yc, feature_cols_c = load_classification_data(horizon)
    split_idx_c = int(len(Xc) * 0.8)
    Xc_train, Xc_test = Xc.iloc[:split_idx_c], Xc.iloc[split_idx_c:]
    yc_train, yc_test = yc.iloc[:split_idx_c], yc.iloc[split_idx_c:]

    clf_model, le = train_xgb_classifier(Xc_train, yc_train)
    clf_metrics = evaluate_classifier(clf_model, le, Xc_test, yc_test)
    print(f"Classification accuracy: {clf_metrics['accuracy']:.4f}, f1: {clf_metrics['f1']:.4f}")
    print(clf_metrics["report"])
    clf_path = save_classifier(clf_model, le, horizon)
    print(f"Saved to: {clf_path}")


if __name__ == "__main__":
    for horizon in HORIZONS:
        train_horizon(horizon)