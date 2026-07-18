"""
Hyperparameter search for the regressor and classifier, using
time-series cross-validation (never testing on data that precedes
training data). Small, explicit grids -- not exhaustive, but each
combination is actually fit and scored, and the chosen winner is
printed with its CV score so it's defensible in a writeup (not just
"we picked good-looking numbers").
"""
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

REGRESSOR_GRID = [
    {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05},
    {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05},
    {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.03},
    {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.02},
    {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1},
]

CLASSIFIER_GRID = [
    {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05},
    {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05},
    {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.03},
    {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.02},
    {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1},
]


def tune_regressor(X, y, n_splits=4, grid=None):
    """
    Fits every candidate in `grid` across TimeSeriesSplit folds,
    returns (best_params, results_df) sorted by mean RMSE (lower is
    better). Import is local to avoid a circular import with
    regression_model.py.
    """
    from models.regression_model import train_xgb_regressor, evaluate_regressor

    grid = grid or REGRESSOR_GRID
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows = []

    for params in grid:
        fold_rmses = []
        for train_idx, test_idx in tscv.split(X):
            model = train_xgb_regressor(X.iloc[train_idx], y.iloc[train_idx], params=params)
            metrics = evaluate_regressor(model, X.iloc[test_idx], y.iloc[test_idx])
            fold_rmses.append(metrics["RMSE"])
        rows.append({**params, "mean_RMSE": sum(fold_rmses) / len(fold_rmses)})

    results = pd.DataFrame(rows).sort_values("mean_RMSE")
    best_params = {k: v for k, v in results.iloc[0].items() if k != "mean_RMSE"}
    best_params = {k: (int(v) if k in ("n_estimators", "max_depth") else v) for k, v in best_params.items()}
    return best_params, results


def tune_classifier(X, y, n_splits=4, grid=None):
    """Same as tune_regressor but scores on mean F1 (higher is better)."""
    from models.classification_model import train_xgb_classifier, evaluate_classifier

    grid = grid or CLASSIFIER_GRID
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows = []

    for params in grid:
        fold_f1s = []
        for train_idx, test_idx in tscv.split(X):
            model, le = train_xgb_classifier(X.iloc[train_idx], y.iloc[train_idx], params=params)
            metrics = evaluate_classifier(model, le, X.iloc[test_idx], y.iloc[test_idx])
            fold_f1s.append(metrics["f1"])
        rows.append({**params, "mean_f1": sum(fold_f1s) / len(fold_f1s)})

    results = pd.DataFrame(rows).sort_values("mean_f1", ascending=False)
    best_params = {k: v for k, v in results.iloc[0].items() if k != "mean_f1"}
    best_params = {k: (int(v) if k in ("n_estimators", "max_depth") else v) for k, v in best_params.items()}
    return best_params, results