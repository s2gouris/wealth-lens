"""
Storage helpers. We use Parquet (columnar, compressed, fast to filter
by column) rather than CSV — this matters once you're loading years
of daily data across dozens of tickers for feature engineering.
"""
import pandas as pd
from pathlib import Path


def save_parquet(df: pd.DataFrame, path: Path, dedupe_on: list[str] | None = None):
    """
    Save a DataFrame to Parquet. If a file already exists at `path`,
    merge new rows in and de-duplicate (so this function is safe to
    call repeatedly for incremental/daily updates without creating
    duplicate rows).
    """
    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, df], ignore_index=True)
        if dedupe_on:
            combined = combined.drop_duplicates(subset=dedupe_on, keep="last")
        combined.to_parquet(path, index=False)
    else:
        df.to_parquet(path, index=False)


def load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)
