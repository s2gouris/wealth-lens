"""
Shared feature column definitions and a coverage-aware selector.

Not all feature groups are populated yet (fundamentals are still
backfilling, news sentiment is partial). Rather than hardcoding one
feature list and having every row get dropped for missing fundamentals,
select_available_features() checks actual null-rate in the data and
only uses columns that are usably populated. As fundamentals/news
backfill over time, re-running training automatically picks them up
with no code changes -- the feature set grows on its own.
"""
import pandas as pd

TECHNICAL_COLS = [
    "return_1d", "return_5d", "return_21d",
    "sma_20", "sma_50", "ema_12", "ema_26",
    "bollinger_pct", "volatility_21d", "volume_avg_21d", "volume_ratio",
]

FUNDAMENTAL_COLS = [
    "eps", "revenue", "profit_margin", "debt_to_equity", "revenue_growth_yoy", "pe_ratio",
]

MACRO_COLS = [
    "fed_funds_rate", "cpi", "unemployment_rate", "yield_curve_10y2y",
]

SENTIMENT_COLS = [
    "news_sentiment", "news_volume",
]

ENGINEERED_COLS = ["rsi_14", "macd_histogram", "atr_14", "dist_from_52w_high", "roc_63"]

ALL_FEATURE_COLS = TECHNICAL_COLS + FUNDAMENTAL_COLS + MACRO_COLS + SENTIMENT_COLS + ENGINEERED_COLS

HORIZONS = ["1m", "3m", "6m"]


def select_available_features(df: pd.DataFrame, min_coverage: float = 0.5) -> list[str]:
    """
    Returns the subset of ALL_FEATURE_COLS that are populated in at
    least `min_coverage` fraction of rows. Technical + macro columns
    are essentially always fully populated once warm-up rows are
    dropped; fundamental/sentiment columns only get included once
    enough tickers have real backfilled data.
    """
    available = []
    for col in ALL_FEATURE_COLS:
        if col not in df.columns:
            continue
        coverage = df[col].notna().mean()
        if coverage >= min_coverage:
            available.append(col)
    return available


def add_ticker_dummies(df: pd.DataFrame, feature_cols: list[str]) -> tuple:
    """
    One-hot encodes ticker as a feature. Without this, the model sees
    all companies' rows mixed together with no way to learn that, say,
    NVDA and KO behave differently -- it's forced to treat every row
    as if it came from a generic, interchangeable stock.
    """
    dummies = pd.get_dummies(df["ticker"], prefix="ticker")
    df_with_dummies = pd.concat([df, dummies], axis=1)
    return df_with_dummies, feature_cols + list(dummies.columns)