"""
Interactive visualizer for HYDROGEL_PACK Round 3 historical data.

Run from repo root:
  streamlit run round3work/hydrogel_work/hydrogel_visualizer.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "Prosperity4Data" / "ROUND_3"
SYMBOL = "HYDROGEL_PACK"
DAYS = [0, 1, 2]


@st.cache_data
def load_prices(day: int) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"prices_round_3_day_{day}.csv", sep=";")
    df = df[df["product"] == SYMBOL].copy()
    df["day"] = day
    df["wall_mid"] = (
        df[["bid_price_1", "bid_price_2", "bid_price_3"]].min(axis=1) +
        df[["ask_price_1", "ask_price_2", "ask_price_3"]].max(axis=1)
    ) / 2.0
    df["best_spread"] = df["ask_price_1"] - df["bid_price_1"]
    df["wall_spread"] = (
        df[["ask_price_1", "ask_price_2", "ask_price_3"]].max(axis=1) -
        df[["bid_price_1", "bid_price_2", "bid_price_3"]].min(axis=1)
    )
    return df.reset_index(drop=True)


@st.cache_data
def load_trades(day: int) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"trades_round_3_day_{day}.csv", sep=";")
    df = df[df["symbol"] == SYMBOL].copy()
    df["day"] = day
    return df.reset_index(drop=True)


def concat_days(days: list[int], concat: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    GAP = 1_100_000  # gap between days when concatenating
    prices_parts, trades_parts = [], []
    for d in days:
        p = load_prices(d).copy()
        t = load_trades(d).copy()
        if concat:
            offset = d * GAP
            p["x"] = p["timestamp"] + offset
            t["x"] = t["timestamp"] + offset
        else:
            p["x"] = p["timestamp"]
            t["x"] = t["timestamp"]
        prices_parts.append(p)
        trades_parts.append(t)
    return pd.concat(prices_parts, ignore_index=True), pd.concat(trades_parts, ignore_index=True)


# ── App ──────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="HYDROGEL_PACK visualizer", layout="wide")
st.title("HYDROGEL_PACK — Round 3 data explorer")

with st.sidebar:
    st.header("Controls")
    days_sel = st.multiselect("Days", DAYS, default=DAYS)
    concat = st.checkbox("Concatenate days on x-axis", value=True)
    show_trades = st.checkbox("Overlay trades", value=True)
    show_wall_mid = st.checkbox("Show wall mid", value=True)
    show_best_bid_ask = st.checkbox("Show best bid / ask", value=True)
    show_book_levels = st.checkbox("Show all 3 book levels", value=False)
    show_spread = st.checkbox("Show spread chart", value=True)
    show_volume = st.checkbox("Show trade volume chart", value=True)
    show_mid_returns = st.checkbox("Show mid-price returns", value=False)

if not days_sel:
    st.warning("Select at least one day.")
    st.stop()

prices, trades = concat_days(days_sel, concat)

DAY_COLORS = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}

# ── 1. Price chart ────────────────────────────────────────────────────────────
st.subheader("Price — mid / bid / ask")
fig = go.Figure()

for d in days_sel:
    p = prices[prices["day"] == d].sort_values("x")
    color = DAY_COLORS[d]
    label = f"day {d}"

    fig.add_trace(go.Scatter(
        x=p["x"], y=p["mid_price"],
        name=f"mid ({label})", mode="lines",
        line=dict(color=color, width=1.5),
    ))

    if show_wall_mid:
        fig.add_trace(go.Scatter(
            x=p["x"], y=p["wall_mid"],
            name=f"wall mid ({label})", mode="lines",
            line=dict(color=color, width=1, dash="dot"),
        ))

    if show_best_bid_ask:
        fig.add_trace(go.Scatter(
            x=p["x"], y=p["bid_price_1"],
            name=f"best bid ({label})", mode="lines",
            line=dict(color=color, width=0.8, dash="dash"),
            opacity=0.6,
        ))
        fig.add_trace(go.Scatter(
            x=p["x"], y=p["ask_price_1"],
            name=f"best ask ({label})", mode="lines",
            line=dict(color=color, width=0.8, dash="dash"),
            opacity=0.6,
            fill="tonexty",
            fillcolor="rgba({},{},{},0.05)".format(*[int(c*255) for c in mcolors.to_rgb(color)]),
        ))

    if show_book_levels:
        for lvl in [2, 3]:
            bc = f"bid_price_{lvl}"
            ac = f"ask_price_{lvl}"
            if bc in p.columns and p[bc].notna().any():
                fig.add_trace(go.Scatter(
                    x=p["x"], y=p[bc],
                    name=f"bid L{lvl} ({label})", mode="lines",
                    line=dict(color=color, width=0.5, dash="dot"),
                    opacity=0.35,
                ))
            if ac in p.columns and p[ac].notna().any():
                fig.add_trace(go.Scatter(
                    x=p["x"], y=p[ac],
                    name=f"ask L{lvl} ({label})", mode="lines",
                    line=dict(color=color, width=0.5, dash="dot"),
                    opacity=0.35,
                ))

if show_trades:
    for d in days_sel:
        t = trades[trades["day"] == d].sort_values("x")
        if t.empty:
            continue
        fig.add_trace(go.Scatter(
            x=t["x"], y=t["price"],
            name=f"trades ({f'day {d}'})",
            mode="markers",
            marker=dict(size=5, symbol="circle-open", color=DAY_COLORS[d]),
            opacity=0.7,
        ))

if concat and len(days_sel) > 1:
    GAP = 1_100_000
    for d in days_sel[1:]:
        fig.add_vline(x=d * GAP, line=dict(color="gray", dash="dash", width=1))

fig.update_layout(
    height=500,
    xaxis_title="timestamp" + (" (concatenated)" if concat else ""),
    yaxis_title="price",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

# ── 2. Spread chart ───────────────────────────────────────────────────────────
if show_spread:
    st.subheader("Quoted spread (ask − bid)")
    fig2 = go.Figure()
    for d in days_sel:
        p = prices[prices["day"] == d].sort_values("x")
        fig2.add_trace(go.Scatter(
            x=p["x"], y=p["best_spread"],
            name=f"best spread day {d}", mode="lines",
            line=dict(color=DAY_COLORS[d], width=1),
        ))
        fig2.add_trace(go.Scatter(
            x=p["x"], y=p["wall_spread"],
            name=f"wall spread day {d}", mode="lines",
            line=dict(color=DAY_COLORS[d], width=1, dash="dot"),
            opacity=0.6,
        ))
    if concat and len(days_sel) > 1:
        for d in days_sel[1:]:
            fig2.add_vline(x=d * GAP, line=dict(color="gray", dash="dash", width=1))
    fig2.update_layout(
        height=300,
        xaxis_title="timestamp",
        yaxis_title="spread",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── 3. Trade volume chart ─────────────────────────────────────────────────────
if show_volume:
    st.subheader("Trade volume")
    fig3 = go.Figure()
    for d in days_sel:
        t = trades[trades["day"] == d].sort_values("x")
        if t.empty:
            continue
        fig3.add_trace(go.Bar(
            x=t["x"], y=t["quantity"],
            name=f"volume day {d}",
            marker_color=DAY_COLORS[d],
            opacity=0.7,
        ))
    if concat and len(days_sel) > 1:
        for d in days_sel[1:]:
            fig3.add_vline(x=d * GAP, line=dict(color="gray", dash="dash", width=1))
    fig3.update_layout(
        height=280,
        xaxis_title="timestamp",
        yaxis_title="quantity",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )
    st.plotly_chart(fig3, use_container_width=True)

# ── 4. Mid-price returns ──────────────────────────────────────────────────────
if show_mid_returns:
    st.subheader("Mid-price tick returns")
    fig4 = go.Figure()
    for d in days_sel:
        p = prices[prices["day"] == d].sort_values("x")
        ret = p["mid_price"].diff()
        fig4.add_trace(go.Scatter(
            x=p["x"], y=ret,
            name=f"Δmid day {d}", mode="lines",
            line=dict(color=DAY_COLORS[d], width=0.8),
        ))
    fig4.add_hline(y=0, line=dict(color="black", width=0.8))
    if concat and len(days_sel) > 1:
        for d in days_sel[1:]:
            fig4.add_vline(x=d * GAP, line=dict(color="gray", dash="dash", width=1))
    fig4.update_layout(
        height=280,
        xaxis_title="timestamp",
        yaxis_title="Δ mid price",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )
    st.plotly_chart(fig4, use_container_width=True)

# ── 5. Summary stats ──────────────────────────────────────────────────────────
st.subheader("Session summary")
rows = []
for d in days_sel:
    p = load_prices(d)
    t = load_trades(d)
    rows.append({
        "day": d,
        "mid min": p["mid_price"].min(),
        "mid max": p["mid_price"].max(),
        "mid mean": round(p["mid_price"].mean(), 2),
        "mid std": round(p["mid_price"].std(), 2),
        "best spread median": round(p["best_spread"].median(), 2),
        "wall spread median": round(p["wall_spread"].median(), 2),
        "n trades": len(t),
        "total volume": int(t["quantity"].sum()) if not t.empty else 0,
        "trade price mean": round(t["price"].mean(), 2) if not t.empty else float("nan"),
    })
st.dataframe(pd.DataFrame(rows).set_index("day"), use_container_width=True)
