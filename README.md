# Wealth Lens: Stock Selection & Portfolio Allocation IDSS

An interactive decision support system that helps self-directed investors decide how to allocate their portfolio, based on predicted returns, risk, and a Buy/Hold/Sell signal per stock, personalized to their risk tolerance, diversification limits, and investment horizon.

Collects prices, fundamentals, macro indicators, and news sentiment for the ticker universe defined in `config.py`, trains prediction models on that data, and serves recommendations through an interactive Streamlit app.

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
   Add `.env` to `.gitignore` before committing. Never push API keys to GitHub.

## Usage, full pipeline, in order

**1. Collect raw data** (run once to build your training set):
```
python collect_initial.py
```

**2. Daily incremental update** (run on a schedule to keep data current):
```
python collect_daily_update.py
```

**3. Build the feature table:**
```
python build_features.py
```

**4. Train the models** (return regressor plus Buy/Hold/Sell classifier, per horizon):
```
python evaluate_all.py
```

**5. Generate final predictions for the app:**
```
python generate_predictions.py
```

**6. Run the app:**
```
streamlit run app.py
```
This opens an interactive dashboard where users adjust risk tolerance, diversification cap, and investment horizon via sliders, and see the recommended allocation, along with per-stock Buy/Hold/Sell signals, a risk-vs-return view, and a portfolio summary, update live.

## Data layout

```
data/
  prices/            one Parquet file per ticker, daily OHLCV
  fundamentals/       dated snapshots (point-in-time, not overwritten)
  macro/              shared macro series, joined by date
  news/               one Parquet file per ticker, sentiment-scored articles
  processed/          engineered feature table plus final predictions
saved_models/          trained regressor and classifier, one set per horizon
```

Parquet was chosen over CSV because it's columnar (fast to load only the columns you need for feature engineering) and compressed (years of daily data across many tickers stays small).

## Model architecture

Two models feed the portfolio optimizer:
- **Regressor** (XGBoost): predicts expected forward return per stock, trained separately per horizon (1m/3m/6m)
- **Classifier** (XGBoost): predicts Buy/Hold/Sell action per stock, reported alongside the allocation as a secondary signal

Risk is computed directly from price history (annualized trailing 21-day volatility) rather than a separately trained model.

## Known constraints (see worksheet Data Collection section)

- **Alpha Vantage free tier**: 25 requests/day, ~5/min. Fundamentals need 3 requests/ticker and news needs 1/ticker, so a full 15-ticker collection needs about 60 requests total, well over the daily cap. A single run of `collect_initial.py` will collect what the budget allows and stop; re-running on subsequent days resumes automatically and skips already-collected tickers.
- **Point-in-time fundamentals**: each fundamentals pull is timestamped (`snapshot_date`) rather than overwriting old values, so that later feature engineering can look up "what was known as of date X" rather than leaking today's restated figures into historical training rows.
- **Survivorship bias**: `STARTER_UNIVERSE` reflects currently-listed companies; delisted/failed companies are not represented in history. Worth flagging as a limitation in the proposal.
- **Return prediction is genuinely hard**: regressor R squared is near zero at short horizons (consistent with published short-horizon return literature), improving toward longer horizons. The classifier's directional accuracy (up to about 56% at 6 months vs. a 33% random baseline) is the stronger, more reliable signal, reported honestly in the model evaluation output rather than hidden.

## Next steps

- Expand `STARTER_UNIVERSE` once fundamentals/news are fully backfilled across multiple collection days
- Wire the classifier's Buy/Hold/Sell signal into the optimizer's allocation math directly (currently shown as a parallel, informational signal alongside the regressor-driven allocation)
- Set up a GitHub Actions cron workflow to run `collect_daily_update.py` automatically
