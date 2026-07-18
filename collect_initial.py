"""
Run this ONCE to build your initial training dataset:
    python collect_initial.py

Pulls 5 years of price history (fast, no key needed), then
fundamentals, macro, and news (which use rate-limited APIs so this
will take a few minutes due to the sleep() calls between requests).
"""
from config import STARTER_UNIVERSE, ALPHA_VANTAGE_DAILY_LIMIT
from collectors.prices import collect_prices
from collectors.benchmarks import collect_benchmarks
from collectors.fundamentals import collect_fundamentals
from collectors.macro import collect_macro
from collectors.news import collect_news
from collectors.rate_limit import RequestBudget

if __name__ == "__main__":
    print("=== Step 1: Price history (yfinance) ===")
    collect_prices(STARTER_UNIVERSE, initial=True)

    print("\n=== Step 2: Benchmark indices (yfinance) ===")
    collect_benchmarks(initial=True)

    print("\n=== Step 3: Macro series (FRED) ===")
    collect_macro()

    # Fundamentals (3 req/ticker) and news (1 req/ticker) share one
    # Alpha Vantage daily budget -- see ALPHA_VANTAGE_DAILY_LIMIT in
    # config.py. Fundamentals goes first since it's a one-time
    # historical backfill; once every ticker has fundamentals saved,
    # collect_fundamentals() skips them instantly and the full budget
    # flows to news automatically on later runs.
    budget = RequestBudget(ALPHA_VANTAGE_DAILY_LIMIT)

    print("\n=== Step 4: Fundamentals (Alpha Vantage) ===")
    collect_fundamentals(STARTER_UNIVERSE, request_budget=budget)

    print("\n=== Step 5: News sentiment (Alpha Vantage) ===")
    collect_news(STARTER_UNIVERSE, request_budget=budget)

    print(f"\nAll done. {budget.remaining} Alpha Vantage requests left in today's budget.")
    print("If fundamentals or news were cut short above, just re-run this script")
    print("tomorrow -- already-collected fundamentals are skipped automatically.")
    print("Check the data/ directory for Parquet files.")
