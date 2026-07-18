"""
Trains the regressor and classifier for all three worksheet horizons
(1m/3m/6m), with:
  - 5 additional engineered technical features (RSI, MACD histogram,
    ATR, 52-week-high distance, 63-day ROC) -- see
    models/engineered_features.py
  - ticker one-hot encoding, so the model can learn company-specific
    behavior instead of treating all 15 stocks as interchangeable
  - hyperparameters chosen by real time-series cross-validated search
    (models/tuning.py), not guessed
  - class-balanced sample weighting on the classifier, since the raw
    label distribution is imbalanced (see models/quantile_labels.py
    for why, and the printed comparison below)
  - both the fixed +-5% label threshold AND the data-driven quantile
    threshold are trained and reported side by side, so the choice
    between them is a documented comparison, not a hidden decision

Run build_features.py first. Prints everything needed to defend
these results in a writeup: which features were used, which
hyperparameters were selected and why (CV scores), and metrics for
both labeling schemes.
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
from models.tuning import tune_regressor, tune_classifier
from models.features import HORIZONS


def train_regressor_for_horizon(horizon: str):
    X, y, feature_cols = load_regression_data(horizon)
    print(f"Regressor features ({len(feature_cols)} total, "
          f"{sum(1 for c in feature_cols if c.startswith('ticker_'))} are ticker dummies)")
    print(f"Rows after dropping incomplete: {len(X)}")

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print("Tuning hyperparameters (time-series CV)...")
    best_params, tuning_results = tune_regressor(X_train, y_train, n_splits=4)
    print(f"Best params: {best_params}")
    print(tuning_results.to_string(index=False))

    reg_model = train_xgb_regressor(X_train, y_train, params=best_params)
    reg_metrics = evaluate_regressor(reg_model, X_test, y_test)
    print(f"Held-out test metrics: {reg_metrics}")
    reg_path = save_regressor(reg_model, horizon)
    print(f"Saved to: {reg_path}")


def train_classifier_for_horizon(horizon: str):
    # --- Fixed +-5% threshold ---
    Xf, yf, feature_cols = load_classification_data(horizon, use_quantile_labels=False)
    split_f = int(len(Xf) * 0.8)
    Xf_train, Xf_test = Xf.iloc[:split_f], Xf.iloc[split_f:]
    yf_train, yf_test = yf.iloc[:split_f], yf.iloc[split_f:]

    print("Tuning classifier hyperparameters (time-series CV, fixed threshold)...")
    best_params, tuning_results = tune_classifier(Xf_train, yf_train, n_splits=4)
    print(f"Best params: {best_params}")
    print(tuning_results.to_string(index=False))

    model_fixed, le_fixed = train_xgb_classifier(Xf_train, yf_train, params=best_params)
    metrics_fixed = evaluate_classifier(model_fixed, le_fixed, Xf_test, yf_test)
    print(f"\n[Fixed +-5% threshold] accuracy={metrics_fixed['accuracy']:.4f}, f1={metrics_fixed['f1']:.4f}")
    print(metrics_fixed["report"])
    save_classifier(model_fixed, le_fixed, horizon, path=f"saved_models/classifier_{horizon}_fixed.joblib")

    # --- Quantile (tertile) threshold ---
    Xq, yq, _ = load_classification_data(horizon, use_quantile_labels=True)
    split_q = int(len(Xq) * 0.8)
    Xq_train, Xq_test = Xq.iloc[:split_q], Xq.iloc[split_q:]
    yq_train, yq_test = yq.iloc[:split_q], yq.iloc[split_q:]

    model_q, le_q = train_xgb_classifier(Xq_train, yq_train, params=best_params)
    metrics_q = evaluate_classifier(model_q, le_q, Xq_test, yq_test)
    print(f"[Quantile (tertile) threshold] accuracy={metrics_q['accuracy']:.4f}, f1={metrics_q['f1']:.4f}")
    print(metrics_q["report"])
    save_classifier(model_q, le_q, horizon, path=f"saved_models/classifier_{horizon}_quantile.joblib")


def train_horizon(horizon: str):
    print(f"\n{'=' * 60}")
    print(f"Horizon: {horizon}")
    print("=" * 60)
    print("\n--- Regressor ---")
    train_regressor_for_horizon(horizon)
    print("\n--- Classifier ---")
    train_classifier_for_horizon(horizon)


if __name__ == "__main__":
    for horizon in HORIZONS:
        train_horizon(horizon)