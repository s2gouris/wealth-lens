"""
Computes forward-looking prediction targets: expected return
(regression) and buy/hold/sell (classification), at each of the three
horizons named in the worksheet's Prediction Task section (1/3/6
months).

These use FUTURE prices by construction -- that's what a label is.
Rows within `horizon` trading days of the end of a ticker's price
history will have NaN labels, since the outcome hasn't happened yet.
Drop those before training; they're fine to keep around for live
inference, there's just nothing to check the prediction against yet.
"""
import pandas as pd

# ~21 trading days/month
HORIZONS = {"1m": 21, "3m": 63, "6m": 126}


def add_labels(df: pd.DataFrame, buy_threshold: float = 0.05,
                sell_threshold: float = -0.05) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    for name, days in HORIZONS.items():
        forward_return = df["close"].shift(-days) / df["close"] - 1
        df[f"forward_return_{name}"] = forward_return
        df[f"label_{name}"] = pd.cut(
            forward_return,
            bins=[-float("inf"), sell_threshold, buy_threshold, float("inf")],
            labels=["sell", "hold", "buy"],
        )
    return df
