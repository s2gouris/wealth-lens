"""
Technical features computed from a single ticker's own OHLCV history.
No look-ahead risk here -- every column is a function of data available
up to and including each row's own date (rolling/ewm windows only look
backward).
"""
import numpy as np
import pandas as pd


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    `df` must be one ticker's price history with columns:
    date, ticker, open, high, low, close, volume.
    """
    df = df.sort_values("date").reset_index(drop=True)

    df["return_1d"] = df["close"].pct_change(1)
    df["return_5d"] = df["close"].pct_change(5)
    df["return_21d"] = df["close"].pct_change(21)

    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()

    bb_mid = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    # position of price within the bands: 0 = at lower band, 1 = at upper band
    band_width = (bb_upper - bb_lower).replace(0, np.nan)
    df["bollinger_pct"] = (df["close"] - bb_lower) / band_width

    df["volatility_21d"] = df["return_1d"].rolling(21).std()
    df["volume_avg_21d"] = df["volume"].rolling(21).mean()
    df["volume_ratio"] = df["volume"] / df["volume_avg_21d"].replace(0, np.nan)

    return df
