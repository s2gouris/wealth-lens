"""
Central configuration: ticker universe, storage paths, API keys.
Keep this the single source of truth so collectors don't hardcode things.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # reads .env file into environment variables

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# --- Storage layout ---
DATA_DIR = Path(__file__).parent / "data"
PRICES_DIR = DATA_DIR / "prices"          # partitioned by ticker
FUNDAMENTALS_DIR = DATA_DIR / "fundamentals"
MACRO_DIR = DATA_DIR / "macro"
NEWS_DIR = DATA_DIR / "news"
BENCHMARKS_DIR = DATA_DIR / "benchmarks"  # index prices (S&P 500, TSX) for performance comparison

for d in [PRICES_DIR, FUNDAMENTALS_DIR, MACRO_DIR, NEWS_DIR, BENCHMARKS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- Ticker universe ---
# Start small while you're building/debugging, then expand.
# For your proposal/demo, 20-50 tickers is plenty; you don't need
# the full S&P 500 to prove the system works.
STARTER_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "JPM", "XOM", "JNJ", "PG", "KO",
    "TSLA", "META", "DIS", "WMT", "V",
]

# --- FRED macro series to track ---
# series_id: human-readable name
FRED_SERIES = {
    "FEDFUNDS": "fed_funds_rate",
    "CPIAUCSL": "cpi",
    "UNRATE": "unemployment_rate",
    "T10Y2Y": "yield_curve_10y2y",
}

# --- Historical pull window for initial training set ---
HISTORY_YEARS = 5

# --- Alpha Vantage free-tier budget (shared by fundamentals + news) ---
# Fundamentals needs 3 requests/ticker (EARNINGS, INCOME_STATEMENT,
# BALANCE_SHEET) and news needs 1 request/ticker, so a 15-ticker
# universe needs 60 requests total -- well over the 25/day free cap.
# Both collectors accept a shared RequestBudget so a single run of
# collect_initial.py never exceeds this, and cut-short work resumes
# automatically on the next run (see collectors/rate_limit.py).
ALPHA_VANTAGE_DAILY_LIMIT = 25
