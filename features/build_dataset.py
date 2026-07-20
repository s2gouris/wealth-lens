"""
Orchestrates the feature engineering + labeling layer: for each
ticker, join technical, fundamental, macro, and sentiment features
onto its price history (all point-in-time correct), attach
forward-looking labels, and concatenate into one training-ready table.

Run via build_features.py at the repo root, after collect_initial.py
has populated data/prices, data/fundamentals, data/macro, data/news.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from config import PRICES_DIR, FUNDAMENTALS_DIR, MACRO_DIR, NEWS_DIR, PROCESSED_DIR
from storage import load_parquet, save_parquet
from features.technical import add_technical_features
from features.fundamentals import join_fundamentals
from features.macro import join_macro
from features.sentiment import join_sentiment
from features.labels import add_labels

OUTPUT_PATH = PROCESSED_DIR / "training_set.parquet"


def build_ticker_features(ticker: str, fundamentals_all: pd.DataFrame,
                           macro: pd.DataFrame) -> pd.DataFrame | None:
    prices = load_parquet(PRICES_DIR / f"{ticker}.parquet")
    if prices.empty:
        print(f"  [WARN] no price data for {ticker}, skipping (run collect_initial.py first)")
        return None

    df = add_technical_features(prices)

    if fundamentals_all.empty:
        fundamentals = fundamentals_all
    else:
        fundamentals = fundamentals_all[fundamentals_all["ticker"] == ticker]
    df = join_fundamentals(df, fundamentals)

    df = join_macro(df, macro)

    news = load_parquet(NEWS_DIR / f"{ticker}_news.parquet")
    df = join_sentiment(df, news)

    df = add_labels(df)
    return df


def build_training_set(tickers: list[str]):
    fundamentals_all = load_parquet(FUNDAMENTALS_DIR / "fundamentals_history.parquet")
    macro = load_parquet(MACRO_DIR / "macro_series.parquet")

    frames = []
    for ticker in tickers:
        print(f"Building features for {ticker}...")
        df = build_ticker_features(ticker, fundamentals_all, macro)
        if df is not None:
            frames.append(df)

    if not frames:
        print("No tickers produced features -- did you run collect_initial.py first?")
        return

    combined = pd.concat(frames, ignore_index=True)
    save_parquet(combined, OUTPUT_PATH, dedupe_on=["ticker", "date"])
    print(f"Done. {len(combined)} rows across {len(frames)} tickers saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    from config import STARTER_UNIVERSE
    build_training_set(STARTER_UNIVERSE)
