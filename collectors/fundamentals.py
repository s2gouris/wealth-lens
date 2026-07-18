"""
Collects company fundamentals via Alpha Vantage's OVERVIEW endpoint.

IMPORTANT rate limit constraint: the free Alpha Vantage tier allows
25 requests/day. With 15 starter tickers this is fine for a one-time
pull, but if you expand your universe you'll need to either:
  (a) spread the pull across multiple days, or
  (b) upgrade to a paid tier (~$50/mo for higher limits), or
  (c) switch to a different fundamentals source (e.g., Financial
      Modeling Prep's free tier, which has a higher daily cap).
State whichever you choose in your Data Collection worksheet section —
this is exactly the kind of "practical constraint" the rubric wants.
"""
import requests
import pandas as pd
import time
import sys
from pathlib import Path
from datetime import date

sys.path.append(str(Path(__file__).parent.parent))
from config import ALPHA_VANTAGE_API_KEY, FUNDAMENTALS_DIR
from storage import save_parquet

BASE_URL = "https://www.alphavantage.co/query"

# Fields we care about from the OVERVIEW response
FIELDS = ["Symbol", "PERatio", "EPS", "DebtToEquityRatio",
          "ProfitMargin", "QuarterlyRevenueGrowthYOY", "MarketCapitalization"]


def fetch_overview(ticker: str) -> dict:
    params = {"function": "OVERVIEW", "symbol": ticker, "apikey": ALPHA_VANTAGE_API_KEY}
    resp = requests.get(BASE_URL, params=params, timeout=15)
    data = resp.json()
    if not data or "Symbol" not in data:
        print(f"  [WARN] no fundamentals returned for {ticker} (rate limited or invalid key)")
        return {}
    return data


def collect_fundamentals(tickers: list[str], sleep_seconds: float = 15.0):
    """
    Pull fundamentals for a list of tickers. `sleep_seconds` throttles
    requests to stay under Alpha Vantage's rate limit (free tier:
    ~5 requests/min). Each snapshot is timestamped with today's date
    so that historical fundamentals accumulate over time as this is
    re-run — giving you a point-in-time record rather than always
    overwriting with "current" values.
    """
    rows = []
    for ticker in tickers:
        print(f"Fetching fundamentals for {ticker}...")
        data = fetch_overview(ticker)
        if not data:
            continue
        row = {f: data.get(f) for f in FIELDS}
        row["snapshot_date"] = date.today().isoformat()
        rows.append(row)
        time.sleep(sleep_seconds)  # respect rate limit

    if rows:
        df = pd.DataFrame(rows)
        path = FUNDAMENTALS_DIR / "fundamentals_snapshots.parquet"
        save_parquet(df, path, dedupe_on=["Symbol", "snapshot_date"])
    print(f"Done. {len(rows)}/{len(tickers)} tickers collected.")


if __name__ == "__main__":
    from config import STARTER_UNIVERSE
    collect_fundamentals(STARTER_UNIVERSE)
