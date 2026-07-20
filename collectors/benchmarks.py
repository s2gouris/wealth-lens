"""
Collects benchmark index prices via yfinance (no API key needed, same
source as prices.py). Needed for the worksheet's Impact Simulation
section, which compares model/portfolio performance against the S&P
500 and TSX Composite.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import BENCHMARKS_DIR
from storage import save_parquet
from collectors.prices import fetch_price_history, fetch_latest_day

# Yahoo Finance ticker: human-readable name
BENCHMARKS = {
    "^GSPC": "sp500",
    "^GSPTSE": "tsx_composite",
}


def collect_benchmarks(initial: bool = True):
    for ticker, name in BENCHMARKS.items():
        print(f"Fetching benchmark {name} ({ticker})...")
        df = fetch_price_history(ticker) if initial else fetch_latest_day(ticker)
        if df.empty:
            continue
        path = BENCHMARKS_DIR / f"{name}.parquet"
        save_parquet(df, path, dedupe_on=["date", "ticker"])
    print(f"Done. {len(BENCHMARKS)} benchmarks processed.")


if __name__ == "__main__":
    collect_benchmarks(initial=True)
