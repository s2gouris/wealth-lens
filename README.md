# Data Collection Pipeline: Stock Selection IDSS

Collects prices, fundamentals, macro indicators, and news sentiment
for the ticker universe defined in `config.py`.

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Get free API keys:
   - Alpha Vantage (fundamentals + news): https://www.alphavantage.co/support/#api-key
   - FRED (macro data): https://fred.stlouisfed.org/docs/api/api_key.html

3. Copy `.env.example` to `.env` and fill in your keys:
   ```
   cp .env.example .env
   ```
   Add `.env` to `.gitignore` before committing — never push API keys to GitHub.

## Usage

**Initial historical pull** (run once to build your training set):
```
python collect_initial.py
```

**Daily incremental update** (run on a schedule):
```
python collect_daily_update.py
```

## Data layout

```
data/
  prices/           one Parquet file per ticker, daily OHLCV
  fundamentals/      dated snapshots (point-in-time, not overwritten)
  macro/             shared macro series, joined by date
  news/              one Parquet file per ticker, sentiment-scored articles
```

Parquet was chosen over CSV because it's columnar (fast to load only
the columns you need for feature engineering) and compressed (years
of daily data across many tickers stays small).

## Running the app
```
streamlit run app.py
```
This opens an interactive dashboard where users adjust risk tolerance,
diversification limits, and investment horizon via sliders, and see
the recommended portfolio allocation update live.

## Known constraints (see worksheet Data Collection section)

- **Alpha Vantage free tier**: 25 requests/day, ~5/min. Fundamentals
  and news collection are throttled with `time.sleep()` to respect this.
  At the current 15-ticker universe this fits in one run; expanding
  the universe will require spreading collection across multiple days
  or upgrading the API tier.
- **Point-in-time fundamentals**: each fundamentals pull is timestamped
  (`snapshot_date`) rather than overwriting old values, so that later
  feature engineering can look up "what was known as of date X" rather
  than leaking today's restated figures into historical training rows.
- **Survivorship bias**: `STARTER_UNIVERSE` reflects currently-listed
  companies; delisted/failed companies are not represented in history.
  Worth flagging as a limitation in the proposal.

## Next steps

- Expand `STARTER_UNIVERSE` once the pipeline is validated on 15 tickers
- Build the feature engineering layer (rolling returns, SMA/EMA,
  volatility) on top of `data/prices/`
- Set up a GitHub Actions cron workflow to run `collect_daily_update.py`
  automatically
