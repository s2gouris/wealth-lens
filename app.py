"""
Wealth Lens -- an intelligent decision-support system (IDSS) for the
self-directed investor. It runs a stock watchlist through three models
(return regressor, buy/hold/sell classifier, growth classifier), turns
that into a personalized allocation, and explains every step.

Run with: streamlit run app.py
"""
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from optimizer import allocate

# --- Signal design language (validated status palette: good/warning/critical) ---
SIGNAL_COLOR = {"buy": "#0ca30c", "hold": "#fab219", "sell": "#d03b3b"}
SIGNAL_SYMBOL = {"buy": "circle", "hold": "diamond", "sell": "square"}
SIGNAL_EMOJI = {"buy": "🟢", "hold": "🟡", "sell": "🔴"}
SIGNAL_LABEL = {"buy": "Buy", "hold": "Hold", "sell": "Sell"}
SIGNAL_ORDER = ["buy", "hold", "sell"]

INK = "#0f1f3d"
SURFACE = "#ffffff"
GRID = "#e6ebf2"

st.set_page_config(page_title="Wealth Lens", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    }
    #MainMenu, footer {visibility: hidden;}
    [data-testid="stDecoration"] {display: none;}
    [data-testid="stHeader"] {background: transparent;}
    .block-container {padding-top: 2.2rem; max-width: 1300px;}

    /* keep the sidebar controls off the left edge */
    [data-testid="stSidebar"] [data-testid="stSidebarContent"],
    [data-testid="stSidebar"] [data-testid="stSidebarUserContent"],
    [data-testid="stSidebar"] .block-container {
        padding-left: 1.4rem; padding-right: 1.1rem;
    }
    [data-testid="stSidebar"] {border-right: 1px solid #e6ebf2;}

    .wl-hero {
        background: linear-gradient(120deg, #0f1f3d 0%, #1d4e89 100%);
        color: #fff; padding: 1.5rem 1.8rem; border-radius: 16px;
        margin-bottom: 1.4rem; box-shadow: 0 8px 24px rgba(15,31,61,0.20);
    }
    .wl-hero h1 {color:#fff; font-size:1.85rem; font-weight:750; margin:0; letter-spacing:-0.01em;}
    .wl-hero p {color:#c8d6ea; margin:.4rem 0 0; font-size:.97rem; max-width:70ch;}

    .wl-section {
        font-size:1.2rem; font-weight:700; color:#0f1f3d;
        margin:1.4rem 0 .2rem; padding-bottom:.4rem; border-bottom:2px solid #e6ebf2;
    }
    .wl-sub {color:#52514e; font-size:.9rem; margin:.15rem 0 .6rem;}

    [data-testid="stMetric"] {
        background:#fff; border:1px solid #e6ebf2; border-radius:12px;
        padding:.85rem 1.05rem; box-shadow:0 1px 3px rgba(15,31,61,0.05);
    }
    [data-testid="stMetricValue"] {font-size:1.55rem; font-weight:750; color:#0f1f3d;}
    [data-testid="stMetricLabel"] {font-weight:600; color:#52514e;}

    .wl-tile {
        background:#fff; border:1px solid #e6ebf2; border-radius:12px;
        padding:.9rem 1.1rem; box-shadow:0 1px 3px rgba(15,31,61,0.05);
    }
    .wl-tile .n {font-size:1.9rem; font-weight:800; line-height:1;}
    .wl-tile .l {font-size:.82rem; font-weight:600; color:#52514e; text-transform:uppercase; letter-spacing:.04em;}

    .wl-card {
        background:#fff; border:1px solid #e6ebf2; border-radius:14px;
        padding:1.2rem 1.4rem; box-shadow:0 2px 10px rgba(15,31,61,0.06);
    }
    .wl-badge {
        display:inline-block; padding:.3rem .95rem; border-radius:999px;
        font-weight:750; font-size:1rem; color:#fff; letter-spacing:.02em;
    }
    .wl-kv {display:flex; justify-content:space-between; padding:.35rem 0; border-bottom:1px solid #f0f2f6;}
    .wl-kv .k {color:#52514e; font-size:.9rem;}
    .wl-kv .v {color:#0f1f3d; font-weight:700; font-size:.95rem;}
    .wl-note {
        background:#f2f6fb; border-left:4px solid #1d4e89; border-radius:8px;
        padding:.8rem 1rem; color:#243b5e; font-size:.92rem; margin-top:.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_real_predictions(horizon_months: int) -> pd.DataFrame:
    """Loads model predictions (return + buy/hold/sell + growth) for the chosen horizon."""
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


def style_fig(fig, height=380):
    fig.update_layout(
        height=height,
        plot_bgcolor=SURFACE, paper_bgcolor=SURFACE,
        font=dict(color=INK, family="-apple-system, Segoe UI, system-ui, sans-serif"),
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor="#c3c2b7")
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor="#c3c2b7")
    return fig


def recommendation_text(row, horizon_months, risk_tolerance) -> str:
    sig, tk = row["signal"], row["ticker"]
    conf, gp = row["confidence"], row["growth_prob"]
    er, risk = row["predicted_return"], row["predicted_risk"]
    profile = "conservative" if risk_tolerance < 0.34 else "aggressive" if risk_tolerance > 0.67 else "balanced"

    lead = {
        "buy": f"The classifier flags **{tk}** as a **Buy** ({conf*100:.0f}% confidence), and the growth model puts the "
               f"probability of a positive {horizon_months}-month return at **{gp*100:.0f}%**.",
        "hold": f"The classifier reads **{tk}** as a **Hold** ({conf*100:.0f}% confidence) — no strong directional edge "
                f"either way over {horizon_months} months.",
        "sell": f"The classifier flags **{tk}** as a **Sell** ({conf*100:.0f}% confidence); the growth model sees only a "
                f"**{gp*100:.0f}%** chance of a positive {horizon_months}-month return.",
    }[sig]

    tail = (f" Expected return is **{er*100:+.1f}%** against annualized volatility of **{risk*100:.0f}%**.")
    if sig == "sell" and profile == "conservative":
        tail += " At your conservative setting the allocator leaves this name out entirely."
    elif sig == "buy" and profile == "aggressive":
        tail += " Your aggressive setting lets this conviction translate into a larger position."
    return lead + tail


# ------------------------------------------------------------------ Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Investor Profile")
    risk_tolerance = st.slider(
        "Risk tolerance", 0.0, 1.0, 0.5, 0.05,
        help="0 = very conservative (avoids Sell-flagged & high-volatility names), 1 = very aggressive",
    )
    diversification_cap = st.slider(
        "Max allocation per stock (%)", 5, 15, 8, 1,
        help="No single stock's allocation will exceed this percentage",
    ) / 100.0
    horizon_months = st.select_slider(
        "Investment horizon (months)", [1, 3, 6, 12], value=3,
        help="Longer horizons carry more model signal; shorter ones pull toward equal-weight",
    )
    st.markdown("---")
    st.markdown("#### How it works")
    st.caption(
        "Three models score every stock on your watchlist:\n\n"
        "**1. Return regressor** — expected forward return.\n\n"
        "**2. Buy/Hold/Sell classifier** — the action signal + confidence.\n\n"
        "**3. Growth classifier** — probability the stock rises.\n\n"
        "The optimizer blends all three with your profile above to size positions."
    )

# ------------------------------------------------------------------ Data
predictions = load_real_predictions(horizon_months)
result = allocate(predictions, risk_tolerance, diversification_cap, horizon_months)

# portfolio-level aggregates
w = result["allocation_pct"] / 100.0
expected_return = float((w * result["predicted_return"]).sum())
expected_risk = float((w * result["predicted_risk"]).sum())
equal_weight_return = float(result["predicted_return"].mean())
ratio = expected_return / expected_risk if expected_risk else 0.0
active_positions = int((result["allocation_pct"] >= 1.0).sum())
top3_weight = float(result["allocation_pct"].nlargest(3).sum())
counts = result["signal"].value_counts()
n_buy, n_hold, n_sell = int(counts.get("buy", 0)), int(counts.get("hold", 0)), int(counts.get("sell", 0))

# ------------------------------------------------------------------ Hero
st.markdown(
    """
    <div class="wl-hero">
      <h1>Wealth Lens</h1>
      <p>An intelligent decision-support system for the self-directed investor, from per-stock
      Buy / Hold / Sell signals through to a personalized, risk-aware portfolio allocation.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ 1. Market signals
st.markdown('<div class="wl-section">1 · Market Signals</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="wl-sub">The buy/hold/sell classifier across your {len(result)}-stock '
    f'watchlist over a {horizon_months}-month horizon.</div>',
    unsafe_allow_html=True,
)

t1, t2, t3, t4 = st.columns(4)
for col, key, label in [(t1, "buy", "Buy signals"), (t2, "hold", "Hold signals"), (t3, "sell", "Sell signals")]:
    n = {"buy": n_buy, "hold": n_hold, "sell": n_sell}[key]
    col.markdown(
        f'<div class="wl-tile" style="border-top:4px solid {SIGNAL_COLOR[key]}">'
        f'<div class="n" style="color:{SIGNAL_COLOR[key]}">{SIGNAL_EMOJI[key]} {n}</div>'
        f'<div class="l">{label}</div></div>',
        unsafe_allow_html=True,
    )
top_pick = result.sort_values("growth_prob", ascending=False).iloc[0]
t4.markdown(
    f'<div class="wl-tile" style="border-top:4px solid #1d4e89">'
    f'<div class="n" style="color:#1d4e89">{top_pick["ticker"]}</div>'
    f'<div class="l">Top conviction · {top_pick["growth_prob"]*100:.0f}% growth</div></div>',
    unsafe_allow_html=True,
)

st.write("")
signals_view = result.copy()
signals_view["Signal"] = signals_view["signal"].map(lambda s: f"{SIGNAL_EMOJI[s]} {SIGNAL_LABEL[s]}")
signals_view = signals_view[[
    "ticker", "Signal", "confidence", "growth_prob",
    "predicted_return", "predicted_risk", "allocation_pct",
]]
st.dataframe(
    signals_view,
    hide_index=True,
    use_container_width=True,
    column_config={
        "ticker": st.column_config.TextColumn("Stock", width="small"),
        "Signal": st.column_config.TextColumn("Signal", width="small"),
        "confidence": st.column_config.ProgressColumn(
            "Confidence", min_value=0.0, max_value=1.0, format="percent"),
        "growth_prob": st.column_config.ProgressColumn(
            "Growth prob.", min_value=0.0, max_value=1.0, format="percent"),
        "predicted_return": st.column_config.NumberColumn("Exp. return", format="percent"),
        "predicted_risk": st.column_config.NumberColumn("Volatility", format="percent"),
        "allocation_pct": st.column_config.NumberColumn("Allocation", format="%.1f%%"),
    },
)

# ------------------------------------------------------------------ 2. Focus on a stock
st.markdown('<div class="wl-section">2 · Focus on a Stock</div>', unsafe_allow_html=True)
st.markdown('<div class="wl-sub">Click on any stock from the drop-down below to see the full recommendation and why.</div>',
            unsafe_allow_html=True)

focus = st.selectbox(
    "Choose a stock", result["ticker"].tolist(),
    index=int(result.reset_index(drop=True)["growth_prob"].idxmax()), label_visibility="collapsed",
)
row = result[result["ticker"] == focus].iloc[0]
sig = row["signal"]

c_left, c_right = st.columns([1, 1])
with c_left:
    st.markdown(
        f'<div class="wl-card">'
        f'<div style="display:flex;align-items:center;gap:.8rem;margin-bottom:.7rem;">'
        f'<span style="font-size:1.5rem;font-weight:800;color:#0f1f3d;">{focus}</span>'
        f'<span class="wl-badge" style="background:{SIGNAL_COLOR[sig]}">{SIGNAL_LABEL[sig].upper()}</span>'
        f'</div>'
        f'<div class="wl-kv"><span class="k">Signal confidence</span><span class="v">{row["confidence"]*100:.0f}%</span></div>'
        f'<div class="wl-kv"><span class="k">Probability of growth</span><span class="v">{row["growth_prob"]*100:.0f}%</span></div>'
        f'<div class="wl-kv"><span class="k">Expected return ({horizon_months}mo)</span><span class="v">{row["predicted_return"]*100:+.1f}%</span></div>'
        f'<div class="wl-kv"><span class="k">Annualized volatility</span><span class="v">{row["predicted_risk"]*100:.0f}%</span></div>'
        f'<div class="wl-kv"><span class="k">Suggested allocation</span><span class="v">{row["allocation_pct"]:.1f}%</span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )
with c_right:
    probs = [row["prob_buy"], row["prob_hold"], row["prob_sell"]]
    fig_p = go.Figure()
    for label, p, key in zip(["Buy", "Hold", "Sell"], probs, SIGNAL_ORDER):
        fig_p.add_trace(go.Bar(
            y=["Class split"], x=[p], name=label, orientation="h",
            marker_color=SIGNAL_COLOR[key], text=f"{p*100:.0f}%", textposition="inside",
            insidetextanchor="middle",
        ))
    fig_p.update_layout(barmode="stack", showlegend=True)
    fig_p.update_xaxes(visible=False, range=[0, 1])
    fig_p.update_yaxes(visible=False)
    st.markdown('<div class="wl-sub" style="margin-bottom:0">Classifier probability split</div>', unsafe_allow_html=True)
    st.plotly_chart(style_fig(fig_p, height=150), use_container_width=True, config={"displayModeBar": False})
    st.markdown(f'<div class="wl-note">{recommendation_text(row, horizon_months, risk_tolerance)}</div>',
                unsafe_allow_html=True)

# ------------------------------------------------------------------ 3. Allocation
st.markdown('<div class="wl-section">3 · Recommended Allocation</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="wl-sub">Adjust your profile in the sidebar and watch it update.</div>',
    unsafe_allow_html=True,
)
alloc_view = result[result["allocation_pct"] >= 0.01].sort_values("allocation_pct", ascending=False)
fig_alloc = px.bar(
    alloc_view, x="ticker", y="allocation_pct", color="signal",
    color_discrete_map=SIGNAL_COLOR,
    category_orders={"signal": SIGNAL_ORDER},
    labels={"ticker": "Stock", "allocation_pct": "Allocation (%)", "signal": "Signal"},
)
fig_alloc.for_each_trace(lambda tr: tr.update(name=SIGNAL_LABEL.get(tr.name, tr.name)))
fig_alloc.update_traces(marker_line_width=0)
st.plotly_chart(style_fig(fig_alloc), use_container_width=True)

# ------------------------------------------------------------------ 4. Risk vs Return
st.markdown('<div class="wl-section">4 · Risk vs. Return</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="wl-sub">Each dot is a stock, sized by its allocation and shaped/colored by its signal. '
    'Up-and-to-the-left is the sweet spot: more return for less risk.</div>',
    unsafe_allow_html=True,
)
scatter = result.copy()
# floor the bubble size so names the allocator avoids (0% weight) still
# appear as the smallest dots -- a 0-sized group otherwise breaks Plotly's
# size scaling. Real allocation stays in the hover.
scatter["bubble"] = scatter["allocation_pct"].clip(lower=0.6)
fig_sc = px.scatter(
    scatter, x="predicted_risk", y="predicted_return",
    size="bubble", color="signal", symbol="signal", text="ticker",
    color_discrete_map=SIGNAL_COLOR, symbol_map=SIGNAL_SYMBOL,
    category_orders={"signal": SIGNAL_ORDER},
    hover_name="ticker",
    hover_data={"bubble": False, "signal": False, "allocation_pct": ":.1f",
                "predicted_risk": ":.1%", "predicted_return": ":.1%"},
    labels={"predicted_risk": "Annualized volatility", "predicted_return": "Expected return",
            "signal": "Signal", "allocation_pct": "Allocation %"},
    size_max=34,
)
fig_sc.for_each_trace(lambda tr: tr.update(name=SIGNAL_LABEL.get(tr.name, tr.name)))
fig_sc.update_traces(textposition="top center", textfont_size=10)
fig_sc.update_xaxes(tickformat=".0%")
fig_sc.update_yaxes(tickformat=".0%")
st.plotly_chart(style_fig(fig_sc, height=440), use_container_width=True)

# ------------------------------------------------------------------ 5. Portfolio summary
st.markdown('<div class="wl-section">5 · Portfolio Summary</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="wl-sub">Based on your portfolio below is your performance overview.</div>',
    unsafe_allow_html=True,
)
m1, m2, m3, m4 = st.columns(4)
m1.metric(
    "Expected return", f"{expected_return*100:.1f}%",
    delta=f"{(expected_return - equal_weight_return)*100:+.1f}% vs equal-weight",
    help="Allocation-weighted expected forward return, vs. an equal-weight watchlist.",
)
m2.metric(
    "Expected volatility", f"{expected_risk*100:.1f}%",
    help="Allocation-weighted annualized volatility.",
)
m3.metric(
    "Return / risk", f"{ratio:.2f}",
    help="Expected return divided by expected volatility — higher is a better risk-adjusted trade-off.",
)
m4.metric(
    "Active positions", f"{active_positions}",
    delta=f"of {len(result)} on watchlist", delta_color="off",
    help="Stocks receiving ≥1% — conservative profiles concentrate into fewer, steadier names.",
)

profile = "conservative" if risk_tolerance < 0.34 else "aggressive" if risk_tolerance > 0.67 else "balanced"
top_holdings = result.nlargest(3, "allocation_pct")["ticker"].tolist()
st.markdown(
    f'<div class="wl-note">At your <b>{profile}</b> profile over <b>{horizon_months} months</b>, '
    f'Wealth Lens concentrates into <b>{active_positions}</b> of {len(result)} names, led by '
    f'<b>{", ".join(top_holdings)}</b>. The classifier currently sees <b>{n_buy} Buy</b>, '
    f'<b>{n_hold} Hold</b>, and <b>{n_sell} Sell</b> signals, and the resulting book targets a '
    f'<b>{expected_return*100:.1f}%</b> return at <b>{expected_risk*100:.1f}%</b> volatility '
    f'(a {ratio:.2f} return/risk ratio, {(expected_return - equal_weight_return)*100:+.1f}% of expected '
    f'return added over equal-weighting).</div>',
    unsafe_allow_html=True,
)

