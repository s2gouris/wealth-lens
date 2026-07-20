"""
Rule-based portfolio allocator.

Turns per-stock model output into portfolio weights. It consumes all
three model signals when they're present, so the allocation reflects
the full pipeline rather than the regressor alone:

  - predicted_return / predicted_risk -> the core risk-adjusted score
  - growth_prob (binary classifier)   -> a conviction tilt: names the
        model is confident will rise get nudged up, and vice versa
  - signal (buy/hold/sell classifier) -> conservative investors drop
        stocks the classifier flags as "sell" outright

All signal columns are optional; if a caller passes a plain
return/risk frame (e.g. the __main__ demo), the allocator falls back
to the original risk-adjusted behavior with no signal tilt.
"""
import numpy as np
import pandas as pd

# how hard growth_prob pulls the score. growth_prob-0.5 spans [-0.5, 0.5],
# so at 0.30 the tilt moves a score by up to +-0.15 -- on the same scale
# as the predicted returns themselves, meaningful but not dominant.
CONVICTION_STRENGTH = 0.30


def allocate(predictions: pd.DataFrame, risk_tolerance: float, diversification_cap: float, horizon_months: int,) -> pd.DataFrame:
    df = predictions.copy().reset_index(drop=True)

    risk_penalty = 3.0 * (1.0 - risk_tolerance)
    df["score"] = df["predicted_return"] - risk_penalty * df["predicted_risk"]

    # Conviction tilt from the growth classifier: lean into names the
    # model is confident about, lean away from the ones it isn't.
    if "growth_prob" in df.columns:
        df["score"] += CONVICTION_STRENGTH * (df["growth_prob"] - 0.5)

    # Eligibility: conservative investors drop their highest-volatility
    # names and anything the classifier flags as "sell". Excluded names
    # are set aside entirely (0 weight) rather than kept in the pool, so
    # the cap redistribution below never has to reason about them.
    eligible = pd.Series(True, index=df.index)
    if risk_tolerance < 0.34:
        cutoff = df["predicted_risk"].quantile(0.6)
        eligible &= df["predicted_risk"] <= cutoff
        if "signal" in df.columns:
            eligible &= df["signal"] != "sell"
    if not eligible.any():  # never leave the investor with nothing
        eligible[:] = True

    act = df[eligible].copy()
    n = len(act)

    # softmax over scores -> a weight per eligible stock
    exp_scores = np.exp(act["score"] - act["score"].max())
    act["weight"] = exp_scores / exp_scores.sum()

    # scale so even short horizons keep meaningful differentiation between
    # names; shorter horizons pull the book toward equal-weight
    horizon_confidence = min(0.5 + horizon_months / 24.0, 1.0)
    equal_weight = 1.0 / n
    act["weight"] = horizon_confidence * act["weight"] + (1 - horizon_confidence) * equal_weight

    # Can't cap below equal weight: with n eligible names, n * cap < 1 is
    # infeasible, so the tightest achievable per-name cap is 1/n.
    effective_cap = max(diversification_cap, 1.0 / n)
    for _ in range(50):
        over_cap = act["weight"] > effective_cap
        if not over_cap.any():
            break
        excess = (act.loc[over_cap, "weight"] - effective_cap).sum()
        act.loc[over_cap, "weight"] = effective_cap
        under_cap = ~over_cap
        under_sum = act.loc[under_cap, "weight"].sum()
        if under_sum <= 0:
            break
        act.loc[under_cap, "weight"] += excess * (act.loc[under_cap, "weight"] / under_sum)

    act["weight"] = act["weight"] / act["weight"].sum()

    df["weight"] = 0.0
    df.loc[act.index, "weight"] = act["weight"]
    df["allocation_pct"] = (df["weight"] * 100).round(2)

    # carry through whichever signal columns were supplied, so the UI
    # can show the model's per-stock reasoning next to each weight
    passthrough = [c for c in ["signal", "confidence", "prob_buy", "prob_hold",
                               "prob_sell", "growth_prob"] if c in df.columns]
    out_cols = ["ticker", "predicted_return", "predicted_risk", "allocation_pct"] + passthrough
    return df[out_cols].sort_values("allocation_pct", ascending=False)


if __name__ == "__main__":
    fake_predictions = pd.DataFrame({
        "ticker": ["AAPL", "MSFT", "TSLA"],
        "predicted_return": [0.08, 0.07, 0.20],
        "predicted_risk": [0.15, 0.14, 0.45],
        "signal": ["buy", "hold", "sell"],
        "growth_prob": [0.72, 0.55, 0.40],
    })
    print(allocate(fake_predictions, risk_tolerance=0.7, diversification_cap=0.20, horizon_months=3))
