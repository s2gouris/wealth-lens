"""
Rule-based portfolio allocator.
"""
import numpy as np
import pandas as pd


def allocate(predictions: pd.DataFrame, risk_tolerance: float, diversification_cap: float) -> pd.DataFrame:
    df = predictions.copy()

    risk_penalty = 3.0 * (1.0 - risk_tolerance)
    df["score"] = df["predicted_return"] - risk_penalty * df["predicted_risk"]

    exp_scores = np.exp(df["score"] - df["score"].max())
    df["weight"] = exp_scores / exp_scores.sum()

    for _ in range(50):
        over_cap = df["weight"] > diversification_cap
        if not over_cap.any():
            break
        excess = (df.loc[over_cap, "weight"] - diversification_cap).sum()
        df.loc[over_cap, "weight"] = diversification_cap
        under_cap = ~over_cap
        df.loc[under_cap, "weight"] += excess * (df.loc[under_cap, "weight"] / df.loc[under_cap, "weight"].sum())

    df["weight"] = df["weight"] / df["weight"].sum()
    df["allocation_pct"] = (df["weight"] * 100).round(2)

    return df[["ticker", "predicted_return", "predicted_risk", "allocation_pct"]].sort_values(
        "allocation_pct", ascending=False
    )