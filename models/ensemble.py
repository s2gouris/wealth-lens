"""
Ensembles the XGBoost and LightGBM models by averaging their
predictions (regression) or their class probabilities
(classification). This is a standard variance-reduction technique --
two models trained differently (level-wise vs leaf-wise tree growth)
tend to make partially independent errors, so averaging their outputs
often reduces overall error even when neither model is individually
best. Whether it actually helps on this data is tested below, not
assumed -- compare_ensemble_regressor/classifier print the blend
against both individual models so the improvement (or lack of one) is
verifiable, not claimed.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_recall_fscore_support, classification_report,
)

from models.regression_model import (
    load_training_data as load_regression_data,
    train_xgb_regressor,
    evaluate_regressor,
)
from models.classification_model import (
    load_training_data as load_classification_data,
    train_xgb_classifier,
)
from models.lightgbm_model import train_lgb_regressor, train_lgb_classifier


def blend_regressor_predictions(xgb_model, lgb_model, X):
    """Simple average of both models' predicted returns."""
    xgb_preds = xgb_model.predict(X)
    lgb_preds = lgb_model.predict(X)
    return (xgb_preds + lgb_preds) / 2


def evaluate_blend_regressor(xgb_model, lgb_model, X_test, y_test):
    preds = blend_regressor_predictions(xgb_model, lgb_model, X_test)
    return {
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "MAE": mean_absolute_error(y_test, preds),
        "R2": r2_score(y_test, preds),
    }


def blend_classifier_probabilities(xgb_model, xgb_le, lgb_model, lgb_le, X):
    """
    Averages class probabilities from both models, then takes the
    argmax as the blended prediction.
    """
    xgb_probs = xgb_model.predict_proba(X)
    lgb_probs_raw = lgb_model.predict_proba(X)

    reorder = [list(lgb_le.classes_).index(c) for c in xgb_le.classes_]
    lgb_probs = lgb_probs_raw[:, reorder]

    avg_probs = (xgb_probs + lgb_probs) / 2
    pred_idx = avg_probs.argmax(axis=1)
    return xgb_le.inverse_transform(pred_idx), avg_probs


def evaluate_blend_classifier(xgb_model, xgb_le, lgb_model, lgb_le, X_test, y_test):
    preds, _ = blend_classifier_probabilities(xgb_model, xgb_le, lgb_model, lgb_le, X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, preds, average="weighted", zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "report": classification_report(y_test, preds, zero_division=0),
    }


def compare_ensemble_regressor(horizon: str = "1m"):
    from models.lightgbm_model import evaluate_lgb_regressor

    X, y, _ = load_regression_data(horizon)
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    xgb_model = train_xgb_regressor(X_train, y_train)
    lgb_model = train_lgb_regressor(X_train, y_train)

    print(f"\n--- Regressor ensemble ({horizon}) ---")
    print(f"XGBoost alone:  {evaluate_regressor(xgb_model, X_test, y_test)}")
    print(f"LightGBM alone: {evaluate_lgb_regressor(lgb_model, X_test, y_test)}")
    print(f"Blend (avg):    {evaluate_blend_regressor(xgb_model, lgb_model, X_test, y_test)}")


def compare_ensemble_classifier(horizon: str = "1m"):
    from models.classification_model import evaluate_classifier

    Xc, yc, _ = load_classification_data(horizon)
    split = int(len(Xc) * 0.8)
    Xc_train, Xc_test = Xc.iloc[:split], Xc.iloc[split:]
    yc_train, yc_test = yc.iloc[:split], yc.iloc[split:]

    xgb_model, xgb_le = train_xgb_classifier(Xc_train, yc_train)
    lgb_model, lgb_le = train_lgb_classifier(Xc_train, yc_train)

    xgb_metrics = evaluate_classifier(xgb_model, xgb_le, Xc_test, yc_test)
    blend_metrics = evaluate_blend_classifier(xgb_model, xgb_le, lgb_model, lgb_le, Xc_test, yc_test)

    print(f"\n--- Classifier ensemble ({horizon}) ---")
    print(f"XGBoost alone: accuracy={xgb_metrics['accuracy']:.4f}, f1={xgb_metrics['f1']:.4f}")
    print(f"Blend (avg):   accuracy={blend_metrics['accuracy']:.4f}, f1={blend_metrics['f1']:.4f}")
    print(blend_metrics["report"])