"""
Collects macroeconomic series from FRED (Federal Reserve Economic Data).
These are not ticker-specific — one shared table used as a feature
joined onto every stock's rows by date.
"""
from fredapi import Fred
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import FRED_API_KEY, FRED_SERIES, MACRO_DIR
from storage import save_parquet


def collect_macro():
    fred = Fred(api_key=FRED_API_KEY)
    frames = []
    for series_id, name in FRED_SERIES.items():
        print(f"Fetching FRED series {series_id} ({name})...")
        series = fred.get_series(series_id)
        df = series.reset_index()
        df.columns = ["date", "value"]
        df["series"] = name
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    path = MACRO_DIR / "macro_series.parquet"
    save_parquet(combined, path, dedupe_on=["date", "series"])
    print(f"Done. {len(FRED_SERIES)} series collected.")


if __name__ == "__main__":
    collect_macro()
