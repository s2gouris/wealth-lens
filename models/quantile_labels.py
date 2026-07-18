"""
Alternative labeling scheme: tertile (33rd/67th percentile) cutoffs
computed directly from the observed forward-return distribution,
instead of build_features.py's fixed +-5% threshold.

Why: the fixed +-5% threshold assumes stock returns are roughly
symmetric around 0 with roughly 5% being a meaningful move in either
direction. In this dataset's training window, the *observed*
distribution isn't symmetric around that particular cutoff -- most
of the "down" tail sits above -5%, so very few rows ever get labeled
"sell", and the classifier collapses to just predicting buy/hold.

Tertile thresholds are re-derived from the data itself: whatever the
actual 33rd/67th percentiles of forward_return are, that's where the
cutoffs go, guaranteeing roughly balanced classes by construction.
This is a standard technique for label imbalance in return-based
classification (equivalent to a rank-based / quantile discretization)
and is easy to justify in a writeup: "thresholds set to produce
balanced tertiles in the training window" is a defensible, stated
methodology choice, not an arbitrary fix.

Trade-off worth stating explicitly: quantile thresholds are fit to
this specific training window and will shift if the window changes
(e.g. a bear market would push the tertile cutoffs down). The fixed
+-5% threshold doesn't have that problem, but suffers the imbalance
seen here instead. Neither is strictly "more correct" -- report both.
"""
import pandas as pd


def add_quantile_labels(df: pd.DataFrame, horizon: str, low_q: float = 0.33,
                         high_q: float = 0.67) -> pd.DataFrame:
    """
    Adds a `label_{horizon}_quantile` column using tertile cutoffs
    computed from that horizon's forward_return column. Does not
    modify or remove the original fixed-threshold label_{horizon}
    column -- both are available so results can be compared.
    """
    df = df.copy()
    return_col = f"forward_return_{horizon}"
    low_cut = df[return_col].quantile(low_q)
    high_cut = df[return_col].quantile(high_q)

    def _label(r):
        if pd.isna(r):
            return None
        if r <= low_cut:
            return "sell"
        if r >= high_cut:
            return "buy"
        return "hold"

    df[f"label_{horizon}_quantile"] = df[return_col].apply(_label)
    return df, low_cut, high_cut