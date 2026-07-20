"""
Additional engineered features, computed on top of what
features/technical.py already provides in training_set.parquet.

Kept as a separate module (rather than editing the teammate's
features/technical.py) so this is clearly scoped as the modeling
side's feature engineering, computed from data already present in
training_set.parquet (open/high/low/close/ema_12/ema_26) with no new
data collection needed.

All five are standard, well-documented technical indicators -- not
arbitrary transforms -- so each is defensible in a writeup:

  - RSI(14): Relative Strength Index, classic momentum oscillator
    (Wilder, 1978). Bounded 0-100; conventionally >70 = overbought,
    <30 = oversold.
  - MACD histogram: MACD line (ema_12 - ema_26, already computed
    upstream) minus its own 9-day EMA signal line. Standard trend-
    momentum indicator; the histogram's sign change is a common
    entry/exit signal.
  - ATR(14): Average True Range (Wilder, 1978), a volatility measure
    that (unlike volatility_21d, which uses close-to-close returns)
    accounts for intraday gaps via high/low/prior-close.
  - dist_from_52w_high: how far current close sits below its trailing
    252-day high, as a fraction. Standard momentum/mean-reversion
    feature (e.g. used in 52-week-high momentum literature).
  - roc_63: 63-trading-day (~1 quarter) rate of change, a longer
    momentum window than the existing return_21d, giving the model a
    slower-moving trend signal alongside the faster ones.

All are computed per-ticker (grouped) and use only backward-looking
windows -- no lookahead, consistent with the point-in-time design of
the rest of the pipeline.
"""
import numpy as np
import pandas as pd


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd_histogram(ema_12: pd.Series, ema_26: pd.Series, signal_period: int = 9) -> pd.Series:
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=signal_period, adjust=False, min_periods=signal_period).mean()
    return macd_line - signal_line


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prior_close = close.shift(1)
    true_range = pd.concat([
        high - low,
        (high - prior_close).abs(),
        (low - prior_close).abs(),
    ], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds rsi_14, macd_histogram, atr_14, dist_from_52w_high, roc_63
    to a multi-ticker DataFrame (as loaded from training_set.parquet).
    Computed per-ticker via an explicit loop (not groupby().apply(),
    which drops the grouping column in newer pandas versions) to
    avoid mixing one company's price history into another's rolling
    windows.
    """
    df = df.sort_values(["ticker", "date"]).copy()

    pieces = []
    for ticker, g in df.groupby("ticker"):
        g = g.copy()
        g["rsi_14"] = _rsi(g["close"])
        g["macd_histogram"] = _macd_histogram(g["ema_12"], g["ema_26"])
        g["atr_14"] = _atr(g["high"], g["low"], g["close"])
        rolling_high_252 = g["close"].rolling(252, min_periods=252).max()
        g["dist_from_52w_high"] = g["close"] / rolling_high_252 - 1
        g["roc_63"] = g["close"].pct_change(63)
        pieces.append(g)

    result = pd.concat(pieces, ignore_index=True)
    return result.sort_values(["ticker", "date"]).reset_index(drop=True)


ENGINEERED_COLS = ["rsi_14", "macd_histogram", "atr_14", "dist_from_52w_high", "roc_63"]