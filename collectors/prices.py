"""
Collects daily OHLCV price data via yfinance (wraps Yahoo Finance,
no API key required). Used for both the initial historical pull and
daily incremental updates.
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import PRICES_DIR, HISTORY_YEARS
from storage import save_parquet


def fetch_price_history(ticker: str, period_years: int = HISTORY_YEARS) -> pd.DataFrame:
    """Pull `period_years` of daily OHLCV history for one ticker."""
    df = yf.download(ticker, period=f"{period_years}y", interval="1d", progress=False)
    if df.empty:
        print(f"  [WARN] no price data returned for {ticker}")
        return df
    df = df.reset_index()
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    df["ticker"] = ticker
    return df[["date", "ticker", "open", "high", "low", "close", "volume"]]


def fetch_latest_day(ticker: str) -> pd.DataFrame:
    """Pull just the most recent trading day — used for daily updates."""
    df = yf.download(ticker, period="5d", interval="1d", progress=False)
    if df.empty:
        return df
    df = df.reset_index().tail(1)
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    df["ticker"] = ticker
    return df[["date", "ticker", "open", "high", "low", "close", "volume"]]


def collect_prices(tickers: list[str], initial: bool = True):
    """
    Collect prices for a list of tickers and save to Parquet,
    one file per ticker (partitioning keeps files small and makes
    per-ticker updates cheap).
    """
    for ticker in tickers:
        print(f"Fetching prices for {ticker}...")
        df = fetch_price_history(ticker) if initial else fetch_latest_day(ticker)
        if df.empty:
            continue
        path = PRICES_DIR / f"{ticker}.parquet"
        save_parquet(df, path, dedupe_on=["date", "ticker"])
    print(f"Done. {len(tickers)} tickers processed.")


if __name__ == "__main__":
    from config import STARTER_UNIVERSE
    collect_prices(STARTER_UNIVERSE, initial=True)
