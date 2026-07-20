"""
Wealth Lens -- interactive portfolio allocation app.
Run with: streamlit run app.py
"""
import pandas as pd
import streamlit as st
import plotly.express as px


from optimizer import allocate

st.set_page_config(page_title="Wealth Lens", layout="wide")
st.markdown(
    """
    <style>
    [data-testid="stDecoration"] {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def load_real_predictions(horizon_months: int) -> pd.DataFrame:
    """Loads model predictions for the chosen horizon."""
    horizon_map = {1: "1m", 3: "3m", 6: "6m", 12: "6m"}  # 12mo slider uses the 6m model
    path = f"data/processed/final_predictions_{horizon_map[horizon_months]}.parquet"
    try:
        return pd.read_parquet(path)
    except FileNotFoundError:
        st.error(
            f"Predictions file not found: {path}. "
            "Run the pipeline first: collect_initial.py → build_features.py → "
            "evaluate_all.py → generate_predictions.py"
        )
        st.stop()


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
        "Max allocation per stock (%)", 5, 20, 12, 1,
        help="No single stock's allocation will exceed this percentage"
    ) / 100.0
with col3:
    horizon_months = st.select_slider(
        "Investment horizon (months)", [1, 3, 6, 12], value=3,
        help="Shorter horizons pull the allocation closer to equal-weight"
    )

predictions = load_real_predictions(horizon_months)
result = allocate(predictions, risk_tolerance, diversification_cap, horizon_months)

left, right = st.columns([2, 1])

with left:
    st.subheader("Recommended Allocation")
    sorted_result = result.sort_values("allocation_pct", ascending=False)
    st.bar_chart(
    sorted_result.set_index("ticker")["allocation_pct"],
    color="#2A7F7E",
    x_label="Stock",
    y_label="Allocation (%)",
)

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

st.subheader("Risk vs. Return")
st.caption("Each dot is a stock, sized by its allocation. Up and to the left = higher return for less risk — these get favored.")
scatter_data = result.rename(columns={
    "predicted_risk": "Risk",
    "predicted_return": "Expected Return",
})
fig = px.scatter(
    scatter_data,
    x="Risk",
    y="Expected Return",
    size="allocation_pct",
    text="ticker",
    hover_name="ticker",
)
fig.update_traces(textposition="top center", marker_color="#2A7F7E")
fig.update_layout(plot_bgcolor="#FAF7F2", paper_bgcolor="#FAF7F2")
st.plotly_chart(fig, use_container_width=True)

top_holdings = result.nlargest(3, "allocation_pct")["ticker"].tolist()
if risk_tolerance < 0.34:
    explanation = (
        f"At your current (conservative) risk tolerance, the portfolio favors "
        f"low-volatility stocks like {', '.join(top_holdings)}. "
        f"Increase risk tolerance to shift weight toward higher-return, higher-risk names."
    )
elif risk_tolerance < 0.67:
    explanation = (
        f"At your current (balanced) risk tolerance, the portfolio spreads weight "
        f"across {', '.join(top_holdings)} and similar names, trading off return against risk. "
        f"Move the slider either way to tilt more conservative or aggressive."
    )
else:
    explanation = (
        f"At your current (aggressive) risk tolerance, the portfolio tilts toward "
        f"higher-return names like {', '.join(top_holdings)}, accepting more volatility. "
        f"Lower risk tolerance to favor steadier stocks."
    )
st.caption(explanation)

st.subheader("Portfolio Summary")
m1, m2, m3 = st.columns(3)
m1.metric("Expected return", f"{expected_return*100:.1f}%")
m2.metric("Portfolio risk", "Low" if expected_risk < 0.15 else "Medium" if expected_risk < 0.25 else "High")
m3.metric("Holdings", len(result))