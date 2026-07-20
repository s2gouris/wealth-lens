"""
Wealth Lens -- interactive portfolio allocation app.
Run with: streamlit run app.py
"""
import pandas as pd
import streamlit as st

from optimizer import allocate

st.set_page_config(page_title="Wealth Lens", layout="wide")


def fake_predictions() -> pd.DataFrame:
    """Placeholder until real model predictions are wired in."""
    return pd.DataFrame({
        "ticker": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM", "XOM", "JNJ",
                   "PG", "KO", "TSLA", "META", "DIS", "WMT", "V"],
        "predicted_return": [0.08, 0.07, 0.06, 0.09, 0.15, 0.05, 0.04, 0.03,
                              0.03, 0.02, 0.20, 0.10, 0.02, 0.03, 0.06],
        "predicted_risk": [0.15, 0.14, 0.16, 0.20, 0.35, 0.12, 0.18, 0.08,
                            0.07, 0.06, 0.45, 0.28, 0.10, 0.05, 0.13],
    })


st.title("Wealth Lens: Portfolio Allocation Assistant")
st.caption(
    "For a self-directed investor deciding how to allocate across their "
    "stock watchlist. Adjust the parameters below and watch the recommended "
    "allocation update live."
)

col1, col2, col3 = st.columns(3)
with col1:
    risk_tolerance = st.slider(
        "Risk tolerance", 0.0, 1.0, 0.5, 0.05,
        help="0 = very conservative, 1 = very aggressive"
    )
with col2:
    diversification_cap = st.slider(
        "Max allocation per stock (%)", 10, 100, 25, 5,
        help="No single stock's allocation will exceed this percentage"
    ) / 100.0
with col3:
    horizon_months = st.select_slider(
        "Investment horizon (months)", [1, 3, 6, 12], value=3,
        help="Shorter horizons pull the allocation closer to equal-weight"
    )

predictions = fake_predictions()
result = allocate(predictions, risk_tolerance, diversification_cap, horizon_months)

left, right = st.columns([2, 1])

with left:
    st.subheader("Recommended Allocation")
    sorted_result = result.sort_values("allocation_pct", ascending=False)
    st.bar_chart(sorted_result.set_index("ticker")["allocation_pct"], color="#1A2540")

with right:
    st.subheader("Details")
    st.dataframe(
        result.rename(columns={
            "predicted_return": "Pred. Return",
            "predicted_risk": "Pred. Risk",
            "allocation_pct": "Allocation %",
        }),
        hide_index=True,
        use_container_width=True,
    )

expected_return = (result["allocation_pct"] / 100 * result["predicted_return"]).sum()
expected_risk = (result["allocation_pct"] / 100 * result["predicted_risk"]).sum()

st.subheader("Portfolio Summary")
m1, m2, m3 = st.columns(3)
m1.metric("Expected return", f"{expected_return*100:.1f}%")
m2.metric("Portfolio risk", "Low" if expected_risk < 0.15 else "Medium" if expected_risk < 0.25 else "High")
m3.metric("Holdings", len(result))