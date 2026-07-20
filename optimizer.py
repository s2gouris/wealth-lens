"""
Rule-based portfolio allocator.
"""
import numpy as np
import pandas as pd


def allocate(predictions: pd.DataFrame, risk_tolerance: float, diversification_cap: float, horizon_months: int,) -> pd.DataFrame:
    df = predictions.copy()

    risk_penalty = 3.0 * (1.0 - risk_tolerance)
    df["score"] = df["predicted_return"] - risk_penalty * df["predicted_risk"]

    if risk_tolerance < 0.34:
        cutoff = df["predicted_risk"].quantile(0.6)
        df.loc[df["predicted_risk"] > cutoff, "score"] = -np.inf

    scores = df["score"].values
    finite_scores = np.where(
        np.isneginf(scores),
        scores[np.isfinite(scores)].min() if np.isfinite(scores).any() else -50,
        scores,
    )

    exp_scores = np.exp(df["score"] - df["score"].max())
    df["weight"] = exp_scores / exp_scores.sum()

    # scale so even short horizons keep meaningful differentiation between
    horizon_confidence = min(0.5 + horizon_months / 24.0, 1.0)
    equal_weight = 1.0 / len(df)
    df["weight"] = horizon_confidence * df["weight"] + (1 - horizon_confidence) * equal_weight

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

if __name__ == "__main__":
    fake_predictions = pd.DataFrame({
        "ticker": ["AAPL", "MSFT", "TSLA"],
        "predicted_return": [0.08, 0.07, 0.20],
        "predicted_risk": [0.15, 0.14, 0.45],
    })
    print(allocate(fake_predictions, risk_tolerance=0.7, diversification_cap=0.20, horizon_months=3))