"""
Collects historical company fundamentals via Alpha Vantage's EARNINGS,
INCOME_STATEMENT, and BALANCE_SHEET endpoints (3 requests/ticker).
Each endpoint returns a full quarterly history in a single call, so —
unlike the OVERVIEW endpoint, which only returns today's snapshot —
this gives fundamentals that vary across the 5-year training window
instead of one static row repeated for every historical price date.

Point-in-time correctness: each quarterly row is tagged with
`reported_date` (from EARNINGS — when the filing was actually made
public), not `fiscal_date_ending` (the period the report covers).
Feature engineering must join price rows to the most recent fundamentals
row where reported_date <= price date, or it will leak information
from before the filing was public.

Rate limit constraint: 3 requests/ticker x 15 tickers = 45 requests,
which exceeds Alpha Vantage's free-tier cap of 25/day (see
ALPHA_VANTAGE_DAILY_LIMIT in config.py). This collector is resumable:
tickers that already have fundamentals saved are skipped on future
runs, and it stops early (printing what's left) once the request
budget passed in is exhausted, so re-running collect_initial.py on a
later day picks up where it left off.
"""
import requests
import pandas as pd
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import ALPHA_VANTAGE_API_KEY, FUNDAMENTALS_DIR
from storage import save_parquet, load_parquet
from collectors.rate_limit import RequestBudget

BASE_URL = "https://www.alphavantage.co/query"
HISTORY_PATH = FUNDAMENTALS_DIR / "fundamentals_history.parquet"
REQUESTS_PER_TICKER = 3


def _get(function: str, symbol: str) -> dict:
    params = {"function": function, "symbol": symbol, "apikey": ALPHA_VANTAGE_API_KEY}
    resp = requests.get(BASE_URL, params=params, timeout=15)
    return resp.json()


def _already_collected(ticker: str) -> bool:
    existing = load_parquet(HISTORY_PATH)
    if existing.empty:
        return False
    return ticker in existing["ticker"].unique()


def fetch_ticker_fundamentals(ticker: str) -> pd.DataFrame:
    """
    Combine EARNINGS (EPS + reported_date), INCOME_STATEMENT (revenue,
    net income), and BALANCE_SHEET (liabilities, equity) into one
    quarterly fundamentals history for `ticker`.

    A short sleep is needed between these 3 calls, not just between
    tickers -- Alpha Vantage's free tier throttles around 5
    requests/minute, and firing 3 calls back-to-back for one ticker
    can trip that even while under the daily cap. A throttled response
    doesn't raise an HTTP error, it just comes back missing the
    expected keys, which silently produced "incomplete fundamentals"
    warnings on every ticker before this fix.
    """
    earnings = _get("EARNINGS", ticker).get("quarterlyEarnings", [])
    time.sleep(13)
    income = _get("INCOME_STATEMENT", ticker).get("quarterlyReports", [])
    time.sleep(13)
    balance = _get("BALANCE_SHEET", ticker).get("quarterlyReports", [])

    if not earnings or not income or not balance:
        print(f"  [WARN] incomplete fundamentals for {ticker} (rate limited or invalid key)")
        return pd.DataFrame()

    income_by_date = {r["fiscalDateEnding"]: r for r in income}
    balance_by_date = {r["fiscalDateEnding"]: r for r in balance}

    rows = []
    for e in earnings:
        fiscal_date = e["fiscalDateEnding"]
        inc = income_by_date.get(fiscal_date)
        bal = balance_by_date.get(fiscal_date)
        if not inc or not bal or not e.get("reportedDate"):
            continue  # only keep quarters where all three endpoints agree

        try:
            revenue = float(inc["totalRevenue"])
            net_income = float(inc["netIncome"])
            liabilities = float(bal["totalLiabilities"])
            equity = float(bal["totalShareholderEquity"])
            eps = float(e["reportedEPS"]) if e.get("reportedEPS") not in (None, "None") else None
        except (KeyError, TypeError, ValueError):
            continue

        rows.append({
            "ticker": ticker,
            "fiscal_date_ending": fiscal_date,
            "reported_date": e["reportedDate"],  # point-in-time availability marker
            "eps": eps,
            "revenue": revenue,
            "profit_margin": net_income / revenue if revenue else None,
            "debt_to_equity": liabilities / equity if equity else None,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values("fiscal_date_ending")
    df["revenue_growth_yoy"] = df["revenue"].pct_change(periods=4)  # same quarter, 1 year back
    return df


def collect_fundamentals(tickers: list[str], sleep_seconds: float = 15.0,
                          request_budget: RequestBudget | None = None):
    """
    Backfill historical fundamentals for `tickers`. Tickers that already
    have data saved are skipped (fundamentals don't need re-pulling once
    collected — see collect_daily_update.py's quarterly cadence note).
    Stops once `request_budget` runs out so this can share a daily
    Alpha Vantage quota with collect_news() in the same run.
    """
    remaining = [t for t in tickers if not _already_collected(t)]
    skipped = len(tickers) - len(remaining)
    if skipped:
        print(f"  {skipped}/{len(tickers)} tickers already have fundamentals, skipping.")

    collected = 0
    for i, ticker in enumerate(remaining):
        if request_budget is not None and not request_budget.has(REQUESTS_PER_TICKER):
            left = remaining[i:]
            print(f"  [BUDGET] stopping fundamentals early — not enough requests left "
                  f"for {len(left)} remaining ticker(s): {', '.join(left)}. "
                  f"Re-run tomorrow to continue (already-collected tickers are skipped).")
            break

        print(f"Fetching fundamentals for {ticker}...")
        df = fetch_ticker_fundamentals(ticker)
        if request_budget is not None:
            request_budget.spend(REQUESTS_PER_TICKER)
        if not df.empty:
            save_parquet(df, HISTORY_PATH, dedupe_on=["ticker", "fiscal_date_ending"])
            collected += 1
        time.sleep(sleep_seconds)

    print(f"Done. {collected}/{len(remaining)} new tickers collected this run.")


if __name__ == "__main__":
    from config import STARTER_UNIVERSE
    collect_fundamentals(STARTER_UNIVERSE)