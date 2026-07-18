"""
Run this ONCE to build your initial training dataset:
    python collect_initial.py

Pulls 5 years of price history (fast, no key needed), then
fundamentals, macro, and news (which use rate-limited APIs so this
will take a few minutes due to the sleep() calls between requests).
"""
from config import STARTER_UNIVERSE
from collectors.prices import collect_prices
from collectors.fundamentals import collect_fundamentals
from collectors.macro import collect_macro
from collectors.news import collect_news

if __name__ == "__main__":
    print("=== Step 1: Price history (yfinance) ===")
    collect_prices(STARTER_UNIVERSE, initial=True)

    print("\n=== Step 2: Macro series (FRED) ===")
    collect_macro()

    print("\n=== Step 3: Fundamentals (Alpha Vantage) ===")
    collect_fundamentals(STARTER_UNIVERSE)

    print("\n=== Step 4: News sentiment (Alpha Vantage) ===")
    collect_news(STARTER_UNIVERSE)

    print("\nAll done. Check the data/ directory for Parquet files.")
