"""
Aggregates per-article news sentiment (collectors/news.py) into a
relevance-weighted daily score, then rolls it up over a trailing
window and joins it onto price history. Only ever uses articles
published on or before each price date, so this is point-in-time
correct by construction -- no asof trickery needed, just a trailing
window ending at "today".
"""
import pandas as pd


def join_sentiment(prices: pd.DataFrame, news: pd.DataFrame, window_days: int = 7) -> pd.DataFrame:
    prices = prices.sort_values("date").copy()
    prices["date"] = pd.to_datetime(prices["date"]).astype("datetime64[ns]")

    if news.empty:
        prices["news_sentiment"] = pd.NA
        prices["news_volume"] = 0
        return prices

    news = news.copy()
    # Alpha Vantage time_published is "YYYYMMDDTHHMMSS"
    news["published_date"] = pd.to_datetime(
        news["published_at"], format="%Y%m%dT%H%M%S", errors="coerce"
    ).dt.floor("D")
    news = news.dropna(subset=["published_date"])
    news["weighted_sentiment"] = news["sentiment_score"] * news["relevance_score"]

    daily = news.groupby("published_date").agg(
        weighted_sum=("weighted_sentiment", "sum"),
        weight_sum=("relevance_score", "sum"),
        article_count=("title", "count"),
    )

    full_range = pd.date_range(daily.index.min(), prices["date"].max(), freq="D")
    daily = daily.reindex(full_range, fill_value=0.0)
    daily.index.name = "date"

    rolled = daily.rolling(f"{window_days}D").sum().reset_index()
    rolled["news_sentiment"] = rolled["weighted_sum"] / rolled["weight_sum"].replace(0, pd.NA)
    rolled["news_volume"] = rolled["article_count"]
    # Same merge_asof dtype requirement as fundamentals.py/macro.py --
    # date_range's output resolution isn't guaranteed to match prices["date"].
    rolled["date"] = rolled["date"].astype("datetime64[ns]")

    return pd.merge_asof(
        prices, rolled[["date", "news_sentiment", "news_volume"]],
        on="date", direction="backward",
    )