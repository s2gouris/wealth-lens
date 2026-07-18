"""
Joins point-in-time fundamentals onto a price history. Uses
merge_asof(direction="backward") on reported_date, so each price row
gets the most recently *published* filing as of that date -- never a
future filing. This is the reason collectors/fundamentals.py stores
reported_date instead of fiscal_date_ending: joining on fiscal period
end would leak a filing's contents into dates before it was public.
"""
import pandas as pd


def join_fundamentals(prices: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    """
    `prices` is one ticker's price history (must include `date`, `close`).
    `fundamentals` is that same ticker's rows from
    fundamentals_history.parquet (reported_date, eps, revenue,
    profit_margin, debt_to_equity, revenue_growth_yoy).
    """
    prices = prices.sort_values("date").copy()
    prices["date"] = pd.to_datetime(prices["date"])

    if fundamentals.empty:
        for col in ["eps", "revenue", "profit_margin", "debt_to_equity", "revenue_growth_yoy"]:
            prices[col] = pd.NA
        prices["pe_ratio"] = pd.NA
        return prices

    fundamentals = fundamentals.sort_values("reported_date").copy()
    fundamentals["reported_date"] = pd.to_datetime(fundamentals["reported_date"])

    merged = pd.merge_asof(
        prices, fundamentals.drop(columns=["ticker"], errors="ignore"),
        left_on="date", right_on="reported_date",
        direction="backward",
    )
    merged["pe_ratio"] = merged["close"] / merged["eps"].where(merged["eps"] > 0)
    return merged
