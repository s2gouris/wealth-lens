"""
Collects news sentiment via Alpha Vantage's NEWS_SENTIMENT endpoint.
This returns a pre-computed sentiment score per article (no need to
build your own NLP sentiment model), tagged to the relevant tickers.

Shares the same rate limit constraint as fundamentals.py — batch
these calls together against your 25/day budget, or space fundamentals
and news collection across different days.
"""
import requests
import pandas as pd
import time
import sys
from pathlib import Path
from datetime import date

sys.path.append(str(Path(__file__).parent.parent))
from config import ALPHA_VANTAGE_API_KEY, NEWS_DIR
from storage import save_parquet

BASE_URL = "https://www.alphavantage.co/query"


def fetch_news_sentiment(ticker: str) -> pd.DataFrame:
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "apikey": ALPHA_VANTAGE_API_KEY,
        "limit": 50,
    }
    resp = requests.get(BASE_URL, params=params, timeout=15)
    data = resp.json()
    feed = data.get("feed", [])
    if not feed:
        print(f"  [WARN] no news returned for {ticker}")
        return pd.DataFrame()

    rows = []
    for item in feed:
        # each article can mention multiple tickers with different relevance;
        # pull out the score specific to this ticker
        ticker_sentiment = next(
            (t for t in item.get("ticker_sentiment", []) if t["ticker"] == ticker), None
        )
        if not ticker_sentiment:
            continue
        rows.append({
            "ticker": ticker,
            "published_at": item.get("time_published"),
            "title": item.get("title"),
            "relevance_score": float(ticker_sentiment["relevance_score"]),
            "sentiment_score": float(ticker_sentiment["ticker_sentiment_score"]),
            "sentiment_label": ticker_sentiment["ticker_sentiment_label"],
            "pulled_date": date.today().isoformat(),
        })
    return pd.DataFrame(rows)


def collect_news(tickers: list[str], sleep_seconds: float = 15.0):
    for ticker in tickers:
        print(f"Fetching news sentiment for {ticker}...")
        df = fetch_news_sentiment(ticker)
        if not df.empty:
            path = NEWS_DIR / f"{ticker}_news.parquet"
            save_parquet(df, path, dedupe_on=["ticker", "title", "published_at"])
        time.sleep(sleep_seconds)
    print("Done.")


if __name__ == "__main__":
    from config import STARTER_UNIVERSE
    collect_news(STARTER_UNIVERSE)
