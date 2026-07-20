"""
Model 3: LightGBM classifier and regressor, as a second gradient-
boosting architecture alongside XGBoost. Both are cited directly in
the worksheet's "Models" reference section
(github.com/lightgbm-org/LightGBM and github.com/dmlc/xgboost), so
comparing them is on-spec, not an arbitrary addition.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_recall_fscore_support, classification_report,
)
from sklearn.utils.class_weight import compute_sample_weight

sys.path.append(str(Path(__file__).parent.parent))
from models.regression_model import load_training_data as load_regression_data
from models.classification_model import load_training_data as load_classification_data


def train_lgb_regressor(X_train, y_train, params=None):
    default_params = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbosity": -1,
    }
    if params:
        default_params.update(params)

    model = lgb.LGBMRegressor(**default_params)
    model.fit(X_train, y_train)
    return model


def train_lgb_classifier(X_train, y_train, params=None, use_class_weights=True):
    default_params = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbosity": -1,
    }
    if params:
        default_params.update(params)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train)
    sample_weight = compute_sample_weight("balanced", y_encoded) if use_class_weights else None

    model = lgb.LGBMClassifier(**default_params)
    model.fit(X_train, y_encoded, sample_weight=sample_weight)
    return model, le


def evaluate_lgb_regressor(model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "MAE": mean_absolute_error(y_test, preds),
        "R2": r2_score(y_test, preds),
    }


def evaluate_lgb_classifier(model, le, X_test, y_test):
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
        "report": classification_report(y_test_encoded, preds, target_names=le.classes_, zero_division=0),
    }


def compare_xgb_vs_lgb(horizon: str = "1m"):
    from models.regression_model import train_xgb_regressor, evaluate_regressor
    from models.classification_model import train_xgb_classifier, evaluate_classifier

    print(f"\n--- Regressor: XGBoost vs LightGBM ({horizon}) ---")
    X, y, _ = load_regression_data(horizon)
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    xgb_model = train_xgb_regressor(X_train, y_train)
    xgb_metrics = evaluate_regressor(xgb_model, X_test, y_test)
    print(f"XGBoost:  {xgb_metrics}")

    lgb_model = train_lgb_regressor(X_train, y_train)
    lgb_metrics = evaluate_lgb_regressor(lgb_model, X_test, y_test)
    print(f"LightGBM: {lgb_metrics}")

    print(f"\n--- Classifier: XGBoost vs LightGBM ({horizon}) ---")
    Xc, yc, _ = load_classification_data(horizon)
    splitc = int(len(Xc) * 0.8)
    Xc_train, Xc_test = Xc.iloc[:splitc], Xc.iloc[splitc:]
    yc_train, yc_test = yc.iloc[:splitc], yc.iloc[splitc:]

    xgb_clf, xgb_le = train_xgb_classifier(Xc_train, yc_train)
    xgb_clf_metrics = evaluate_classifier(xgb_clf, xgb_le, Xc_test, yc_test)
    print(f"XGBoost:  accuracy={xgb_clf_metrics['accuracy']:.4f}, f1={xgb_clf_metrics['f1']:.4f}")

    lgb_clf, lgb_le = train_lgb_classifier(Xc_train, yc_train)
    lgb_clf_metrics = evaluate_lgb_classifier(lgb_clf, lgb_le, Xc_test, yc_test)
    print(f"LightGBM: accuracy={lgb_clf_metrics['accuracy']:.4f}, f1={lgb_clf_metrics['f1']:.4f}")


def save_model(model, model_type: str, horizon: str = "1m", le=None, path: str | None = None):
    path = path or f"saved_models/lgb_{model_type}_{horizon}.joblib"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if le is not None:
        joblib.dump({"model": model, "label_encoder": le}, path)
    else:
        joblib.dump(model, path)
    return path