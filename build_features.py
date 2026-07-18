"""
Run this after collect_initial.py (and once fundamentals/news have
backfilled enough history) to build the point-in-time feature matrix
and labels used to train the regressor and classifier described in
the worksheet:
    python build_features.py

Output: data/processed/training_set.parquet -- one row per
(ticker, date), with technical/fundamental/macro/sentiment features
and forward_return_{1m,3m,6m} / label_{1m,3m,6m} targets.
"""
from config import STARTER_UNIVERSE
from features.build_dataset import build_training_set

if __name__ == "__main__":
    build_training_set(STARTER_UNIVERSE)
