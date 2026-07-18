"""
Replaces the misleading single 80/20 holdout evaluation with proper
time-series cross-validation for every model. A single fixed split
can land on an unrepresentative time period (as it did for the
growth classifier -- ROC-AUC looked like 0.44 on one split, but
averaged 0.67 across 5 time-ordered folds). CV mean +- std across
folds is the number to report and defend.

After evaluating honestly via CV, the FINAL deployed model for each
target is retrained on ALL available data (not just 80%) -- CV was
only for measuring how well the approach generalizes; once that's
established, there's no reason to withhold 20% of real data from the
model you actually ship.

Run this instead of train.py's single-split reporting.
"""
import numpy as np
import pandas as pd

from models.regression_model import (
    load_training_data as load_regression_data,
    train_xgb_regressor,
    time_series_cv_evaluate as cv_regressor,
    save_model as save_regressor,
)
from models.classification_model import (
    load_training_data as load_classification_data,
    train_xgb_classifier,
    evaluate_classifier,
    save_model as save_classifier,
)
from models.growth_classifier import (
    load_growth_data,
    train_growth_classifier,
    time_series_cv_evaluate as cv_growth,
    save_model as save_growth,
)
from models.tuning import tune_regressor, tune_classifier
from models.features import HORIZONS
from sklearn.model_selection import TimeSeriesSplit


def _classifier_cv_with_params(X, y, params, n_splits=5):
    """
    classification_model.time_series_cv_evaluate() ignores any params
    passed to it and always trains with XGBoost defaults -- this
    inlines the same CV loop but actually uses the tuned params, so
    tuning isn't silently wasted.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []
    for train_idx, test_idx in tscv.split(X):
        model, le = train_xgb_classifier(X.iloc[train_idx], y.iloc[train_idx], params=params)
        metrics = evaluate_classifier(model, le, X.iloc[test_idx], y.iloc[test_idx])
        metrics.pop("report")
        results.append(metrics)
    return pd.DataFrame(results)


def evaluate_and_train_regressor(horizon: str, n_splits: int = 5):
    X, y, feature_cols = load_regression_data(horizon)

    print(f"\n[Regressor - {horizon}]")
    best_params, _ = tune_regressor(X, y, n_splits=4)

    cv_results = cv_regressor(X, y, model_fn=lambda Xt, yt: train_xgb_regressor(Xt, yt, params=best_params), n_splits=n_splits)
    print(f"CV results across {n_splits} folds:")
    print(cv_results.to_string(index=False))
    print(f"Mean RMSE={cv_results['RMSE'].mean():.4f} (+-{cv_results['RMSE'].std():.4f}), "
          f"Mean R2={cv_results['R2'].mean():.4f} (+-{cv_results['R2'].std():.4f})")

    # Final model: trained on ALL data, using the CV-selected hyperparameters
    final_model = train_xgb_regressor(X, y, params=best_params)
    path = save_regressor(final_model, horizon)
    print(f"Final model (trained on all {len(X)} rows) saved to: {path}")
    return cv_results


def evaluate_and_train_classifier(horizon: str, n_splits: int = 5):
    X, y, feature_cols = load_classification_data(horizon, use_quantile_labels=False)

    print(f"\n[Classifier (3-class) - {horizon}]")
    best_params, _ = tune_classifier(X, y, n_splits=4)

    cv_results = _classifier_cv_with_params(X, y, best_params, n_splits=n_splits)
    print(f"CV results across {n_splits} folds:")
    print(cv_results.to_string(index=False))
    print(f"Mean accuracy={cv_results['accuracy'].mean():.4f} (+-{cv_results['accuracy'].std():.4f}), "
          f"Mean f1={cv_results['f1'].mean():.4f} (+-{cv_results['f1'].std():.4f})")

    final_model, final_le = train_xgb_classifier(X, y, params=best_params)
    path = save_classifier(final_model, final_le, horizon)
    print(f"Final model (trained on all {len(X)} rows) saved to: {path}")
    return cv_results


def evaluate_and_train_growth(horizon: str, n_splits: int = 5):
    X, y, feature_cols = load_growth_data(horizon)

    print(f"\n[Growth classifier (binary) - {horizon}]")
    cv_results = cv_growth(X, y, n_splits=n_splits)
    print(f"CV results across {n_splits} folds:")
    print(cv_results.to_string(index=False))
    print(f"Mean accuracy={cv_results['accuracy'].mean():.4f} (+-{cv_results['accuracy'].std():.4f}), "
          f"Mean ROC-AUC={cv_results['roc_auc'].mean():.4f} (+-{cv_results['roc_auc'].std():.4f})")

    final_model = train_growth_classifier(X, y)
    path = save_growth(final_model, horizon)
    print(f"Final model (trained on all {len(X)} rows) saved to: {path}")
    return cv_results


if __name__ == "__main__":
    summary = []
    for horizon in HORIZONS:
        print(f"\n{'=' * 60}")
        print(f"Horizon: {horizon}")
        print("=" * 60)

        reg_cv = evaluate_and_train_regressor(horizon)
        clf_cv = evaluate_and_train_classifier(horizon)
        growth_cv = evaluate_and_train_growth(horizon)

        summary.append({
            "horizon": horizon,
            "regressor_R2": reg_cv["R2"].mean(),
            "classifier_accuracy": clf_cv["accuracy"].mean(),
            "growth_roc_auc": growth_cv["roc_auc"].mean(),
        })

    print(f"\n{'=' * 60}")
    print("SUMMARY (mean across CV folds, all models retrained on full data)")
    print("=" * 60)
    print(pd.DataFrame(summary).to_string(index=False))