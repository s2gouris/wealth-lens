"""
Joins shared macro series (FRED) onto a price history. Series are
pivoted long-to-wide, then merge_asof'd so each price date picks up
the most recently released value and forward-fills between releases
(e.g. a monthly CPI print applies until the next one is out).

Known limitation: FRED's `date` index is the reference period (e.g.
CPI "as of" the 1st of the month it covers), not the actual publication
date, which typically lags 2-4 weeks. Unlike fundamentals.py's
reported_date, this join doesn't account for that lag, so macro
features can be a few weeks "early" relative to what was truly public
knowledge at each price date. Worth noting as a limitation rather than
fixing here -- correcting it needs FRED's ALFRED vintage endpoint,
which is a separate (paid-tier-adjacent) data source.
"""
import pandas as pd


def join_macro(prices: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    prices = prices.sort_values("date").copy()
    prices["date"] = pd.to_datetime(prices["date"]).astype("datetime64[ns]")
  
    if macro.empty:
        return prices

    macro = macro.copy()
    macro["date"] = pd.to_datetime(macro["date"]).astype("datetime64[ns]")    
    wide = (
        macro.pivot_table(index="date", columns="series", values="value")
        .sort_index()
        .ffill()
        .reset_index()
    )

    return pd.merge_asof(prices, wide, on="date", direction="backward")
