"""
Interactive multi-voucher / multi-day explorer for pervoucher_analysis outputs.

Requires: pip install streamlit plotly

Run from repo root:
  streamlit run round3work/plotting/original_method/pervoucher_analysis/interactive_visualizer.py

Or from round3work/plotting:
  streamlit run original_method/pervoucher_analysis/interactive_visualizer.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_ROOT = Path(__file__).resolve().parent
_COMBINED = _ROOT.parent / "combined_analysis"
for _p in (_COMBINED, _ROOT.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from frankfurt_style_plots import VOUCHERS, load_book_product, time_x_axis
from plot_iv_smile_round3 import dte_from_csv_day, load_day_wide

PERVOUCHER_ROOT = _ROOT
X_SEGMENT = 55_000.0
ZOOM_START_DEFAULT = 4000
ZOOM_LEN_DEFAULT = 51


def _session_x(df: pd.DataFrame) -> pd.Series:
    """Frankfurt-style 0..50k index within each (day, voucher) group."""
    out = np.zeros(len(df), dtype=float)
    for key, g in df.groupby(["day_label", "voucher"], sort=False):
        ts = np.sort(g["timestamp"].unique())
        mp = dict(zip(ts, time_x_axis(len(ts))))
        out[g.index] = g["timestamp"].map(mp).to_numpy()
    return pd.Series(out, index=df.index)


def load_resdf_slices(vouchers: list[str], days: list[int], concat_days: bool) -> pd.DataFrame:
    parts = []
    for v in vouchers:
        for d in days:
            p = PERVOUCHER_ROOT / v / f"day{d}" / "resdf_slice.csv"
            if not p.exists():
                continue
            sub = pd.read_csv(p)
            sub["voucher"] = v
            sub["day_label"] = d
            sub["log_K_over_S"] = np.log(sub["K"].astype(float) / sub["S"].astype(float))
            parts.append(sub)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df["x_session"] = _session_x(df)
    if concat_days and len(days) > 1:
        df["x_plot"] = df["x_session"] + df["day_label"].astype(float) * X_SEGMENT
    else:
        df["x_plot"] = df["x_session"]
    return df


def atm_ranking_by_day(days: list[int]) -> pd.DataFrame:
    """Rank strikes by mean |log(K/S)| over the session (smaller = closer to ATM on average)."""
    rows = []
    for d in days:
        try:
            w = load_day_wide(d).sort_index()
        except Exception:
            continue
        S = w["S"].to_numpy(dtype=float)
        for v in VOUCHERS:
            if v not in w.columns:
                continue
            K = float(v.split("_")[1])
            m = np.log(K / S)
            m = m[np.isfinite(m)]
            if len(m) == 0:
                continue
            rows.append(
                {
                    "day": d,
                    "voucher": v,
                    "mean_abs_log_KS": float(np.nanmean(np.abs(m))),
                    "mean_log_KS": float(np.nanmean(m)),
                    "median_S": float(np.nanmedian(S)),
                }
            )
    r = pd.DataFrame(rows)
    if r.empty:
        return r
    return r.sort_values(["day", "mean_abs_log_KS"])


def underlying_stats(days: list[int]) -> pd.DataFrame:
    rows = []
    for d in days:
        try:
            w = load_day_wide(d).sort_index()
        except Exception:
            continue
        S = w["S"].astype(float)
        r = np.diff(np.log(S.to_numpy()))
        r = r[np.isfinite(r)]
        rows.append(
            {
                "day": d,
                "median_S": float(S.median()),
                "S_range": float(S.max() - S.min()),
                "ret_std": float(np.nanstd(r)) if len(r) > 1 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def stats_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    def lag1(s: pd.Series) -> float:
        s = s.dropna()
        if len(s) < 3:
            return float("nan")
        a = s.to_numpy(float)
        x, y = a[:-1], a[1:]
        if np.std(x) < 1e-14 or np.std(y) < 1e-14:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    rows = []
    for (v, d), g in df.groupby(["voucher", "day_label"]):
        rows.append(
            {
                "voucher": v,
                "day": int(d),
                "lag1_price_dev": lag1(g["price_dev"]),
                "lag1_iv_res": lag1(g["iv_res"]),
                "mean_abs_price_dev": float(np.nanmean(np.abs(g["price_dev"]))),
                "mean_abs_iv_res": float(np.nanmean(np.abs(g["iv_res"]))),
                "mean_iv_res": float(np.nanmean(g["iv_res"])),
                "n": len(g),
            }
        )
    return pd.DataFrame(rows)


def run() -> None:
    import streamlit as st

    st.set_page_config(page_title="Round 3 pervoucher viz", layout="wide")
    st.title("Round 3 — pervoucher interactive visualizer")
    st.caption(
        "Data: `pervoucher_analysis/<VEV_K>/day*/resdf_slice.csv` + book CSVs for spread / Fig 7. "
        "Model matches `no_wind_down/combined_analysis` (calendar DTE, no intraday wind; smile in log(S/K))."
    )

    with st.sidebar:
        vouchers_sel = st.multiselect("Vouchers", VOUCHERS, default=["VEV_5000", "VEV_5100"])
        days_sel = st.multiselect("Days (CSV day 0/1/2)", [0, 1, 2], default=[0, 1])
        concat = st.checkbox("Concatenate days on x-axis (50k gap)", value=False)
        show_S = st.checkbox("Overlay / twin: underlying S (VELVETFRUIT_EXTRACT)", value=True)
        plot_choice = st.multiselect(
            "Plots",
            [
                "6b IV residual vs time",
                "6c price deviation vs time",
                "IV vs IV_fit vs time",
                "log(K/S) vs time",
                "Spread vs time",
                "Hist price_dev",
                "Hist iv_res",
                "Fig 7a bid/ask/theo/mid (zoom)",
                "Fig 7b normalized (zoom)",
                "resdf table sample",
                "Stats summary table",
            ],
            default=["6b IV residual vs time", "6c price deviation vs time"],
        )
        z_start = st.number_input("7a/7b zoom start row", min_value=0, value=ZOOM_START_DEFAULT, step=100)
        z_len = st.number_input("7a/7b zoom length", min_value=10, value=ZOOM_LEN_DEFAULT, step=1)

    if not vouchers_sel or not days_sel:
        st.warning("Select at least one voucher and one day.")
        return

    if len(days_sel) > 1 and not concat:
        st.info(
            "Multiple days selected with **concatenate** off — traces share the same 0–50k session "
            "x-axis (overlaid). Turn **Concatenate days on x-axis** on to separate days."
        )

    df = load_resdf_slices(vouchers_sel, days_sel, concat)
    if df.empty:
        st.error("No `resdf_slice.csv` found for that selection. Run `build_pervoucher.py` first.")
        return

    st.subheader("Closest to the money (by day)")
    st.caption("Ranked by **mean |log(K/S)|** over the session (smaller ≈ closer to ATM on average).")
    atm = atm_ranking_by_day(days_sel)
    if not atm.empty:
        for d in sorted(atm["day"].unique()):
            sub = atm[atm["day"] == d].copy()
            top = sub.nsmallest(5, "mean_abs_log_KS")[
                ["voucher", "mean_abs_log_KS", "mean_log_KS", "median_S"]
            ]
            st.markdown(f"**Day {d}** (DTE open {dte_from_csv_day(d)})")
            st.dataframe(top.reset_index(drop=True), use_container_width=True)
    else:
        st.info("Could not load `load_day_wide` for ATM table (check Prosperity4Data paths).")

    st.subheader("Underlying session stats")
    us = underlying_stats(days_sel)
    if not us.empty:
        st.dataframe(us, use_container_width=True)

    if "6b IV residual vs time" in plot_choice:
        fig = go.Figure()
        for v in vouchers_sel:
            g = df[df["voucher"] == v]
            if g.empty:
                continue
            for d in sorted(g["day_label"].unique()):
                gg = g[g["day_label"] == d].sort_values("x_plot")
                lab = f"{v} day{d}" if not concat or len(days_sel) == 1 else f"{v} (day {d})"
                fig.add_trace(
                    go.Scatter(x=gg["x_plot"], y=gg["iv_res"], name=lab, mode="lines", line=dict(width=1.1))
                )
        fig.add_hline(y=0, line_dash="solid", line_color="#333")
        fig.update_layout(
            title="Fig 6b — IV residual vs time",
            xaxis_title="Session index" + (" (concatenated)" if concat else ""),
            yaxis_title=r"IV − smile fit",
            height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        if show_S:
            u = df.groupby("x_plot", as_index=False)["S"].mean()
            fig.add_trace(
                go.Scatter(
                    x=u["x_plot"],
                    y=u["S"],
                    name="S (underlying)",
                    yaxis="y2",
                    line=dict(color="gray", dash="dot", width=1),
                )
            )
            fig.update_layout(
                yaxis2=dict(title="S", overlaying="y", side="right", showgrid=False),
            )
        st.plotly_chart(fig, use_container_width=True)

    if "6c price deviation vs time" in plot_choice:
        fig = go.Figure()
        for v in vouchers_sel:
            g = df[df["voucher"] == v]
            for d in sorted(g["day_label"].unique()):
                gg = g[g["day_label"] == d].sort_values("x_plot")
                lab = f"{v} day{d}"
                fig.add_trace(go.Scatter(x=gg["x_plot"], y=gg["price_dev"], name=lab, mode="lines", line=dict(width=1.1)))
        fig.add_hline(y=0, line_color="#333")
        fig.update_layout(
            title="Fig 6c — mid − BS(smile IV)",
            xaxis_title="Session index",
            yaxis_title="price deviation",
            height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        if show_S:
            u = df.groupby("x_plot", as_index=False)["S"].mean()
            fig.add_trace(
                go.Scatter(x=u["x_plot"], y=u["S"], name="S", yaxis="y2", line=dict(color="gray", dash="dot"))
            )
            fig.update_layout(yaxis2=dict(title="S", overlaying="y", side="right", showgrid=False))
        st.plotly_chart(fig, use_container_width=True)

    if "IV vs IV_fit vs time" in plot_choice:
        fig = go.Figure()
        for v in vouchers_sel:
            g = df[df["voucher"] == v]
            for d in sorted(g["day_label"].unique()):
                gg = g[g["day_label"] == d].sort_values("x_plot")
                fig.add_trace(
                    go.Scatter(x=gg["x_plot"], y=gg["iv"], name=f"{v} day{d} IV", mode="lines", line=dict(width=1))
                )
                fig.add_trace(
                    go.Scatter(
                        x=gg["x_plot"],
                        y=gg["iv_fit"],
                        name=f"{v} day{d} IV̂",
                        mode="lines",
                        line=dict(width=1, dash="dash"),
                    )
                )
        fig.update_layout(title="IV vs smile-fitted IV", height=460, legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig, use_container_width=True)

    if "log(K/S) vs time" in plot_choice:
        fig = go.Figure()
        for v in vouchers_sel:
            g = df[df["voucher"] == v]
            for d in sorted(g["day_label"].unique()):
                gg = g[g["day_label"] == d].sort_values("x_plot")
                fig.add_trace(
                    go.Scatter(x=gg["x_plot"], y=gg["log_K_over_S"], name=f"{v} day{d}", mode="lines", line=dict(width=1))
                )
        fig.add_hline(y=0, line_color="black", line_dash="dot")
        fig.update_layout(title="log(K/S) vs time", height=400)
        if show_S:
            u = df.groupby("x_plot", as_index=False)["S"].mean()
            fig.add_trace(go.Scatter(x=u["x_plot"], y=u["S"], name="S", yaxis="y2", line=dict(color="gray")))
            fig.update_layout(yaxis2=dict(title="S", overlaying="y", side="right"))
        st.plotly_chart(fig, use_container_width=True)

    if "Spread vs time" in plot_choice:
        fig = make_subplots(
            rows=len(vouchers_sel),
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=[f"{v} — ask−bid" for v in vouchers_sel],
        )
        for ri, v in enumerate(vouchers_sel, start=1):
            for d in days_sel:
                try:
                    book = load_book_product(d, v)
                except Exception as e:
                    fig.add_annotation(row=ri, col=1, text=f"Book load error: {e}", showarrow=False)
                    continue
                book = book.sort_values("timestamp")
                ts = book["timestamp"].to_numpy()
                off = d * X_SEGMENT if concat and len(days_sel) > 1 else 0
                xb = time_x_axis(len(ts)) + off
                spread = (book["ask_price_1"] - book["bid_price_1"]).astype(float)
                fig.add_trace(
                    go.Scatter(x=xb, y=spread, name=f"{v} d{d}", mode="lines", line=dict(width=1)),
                    row=ri,
                    col=1,
                )
        fig.update_layout(height=min(900, 200 + 180 * len(vouchers_sel)), title_text="Quoted spread vs session time")
        st.plotly_chart(fig, use_container_width=True)

    if "Hist price_dev" in plot_choice:
        fig = go.Figure()
        for v in vouchers_sel:
            g = df[df["voucher"] == v]
            for d in sorted(g["day_label"].unique()):
                fig.add_trace(
                    go.Histogram(
                        x=g[g["day_label"] == d]["price_dev"],
                        name=f"{v} day{d}",
                        opacity=0.55,
                        nbinsx=40,
                    )
                )
        fig.update_layout(barmode="overlay", title="Histogram — price deviation", height=400)
        st.plotly_chart(fig, use_container_width=True)

    if "Hist iv_res" in plot_choice:
        fig = go.Figure()
        for v in vouchers_sel:
            g = df[df["voucher"] == v]
            for d in sorted(g["day_label"].unique()):
                fig.add_trace(
                    go.Histogram(
                        x=g[g["day_label"] == d]["iv_res"],
                        name=f"{v} day{d}",
                        opacity=0.55,
                        nbinsx=40,
                    )
                )
        fig.update_layout(barmode="overlay", title="Histogram — IV residual", height=400)
        st.plotly_chart(fig, use_container_width=True)

    def _zoom_book(v: str, d: int):
        g = df[(df["voucher"] == v) & (df["day_label"] == d)].sort_values("timestamp")
        if g.empty:
            return None
        theo_map = g.set_index("timestamp")["theoretical_mid"]
        try:
            book = load_book_product(int(d), v).sort_values("timestamp")
        except Exception:
            return None
        book = book.copy()
        book["theo"] = book["timestamp"].map(theo_map)
        z = book.iloc[int(z_start) : int(z_start) + int(z_len)]
        if len(z) < 5:
            return None
        tix = np.arange(len(z))
        return tix, z

    if "Fig 7a bid/ask/theo/mid (zoom)" in plot_choice:
        st.caption("Fig 7a — bid/ask / theoretical / mid (zoom window on full book).")
        panels = []
        for v in vouchers_sel:
            for d in days_sel:
                zb = _zoom_book(v, d)
                if zb is None:
                    continue
                tix, z = zb
                panels.append((f"{v} day{d}", tix, z))
        if panels:
            fig7a = make_subplots(
                rows=len(panels), cols=1, subplot_titles=[p[0] for p in panels], vertical_spacing=0.05
            )
            for i, (_, tix, z) in enumerate(panels, start=1):
                bid = z["bid_price_1"].astype(float)
                ask = z["ask_price_1"].astype(float)
                mid = z["mid_price"].astype(float)
                theo = z["theo"].astype(float)
                fig7a.add_trace(go.Scatter(x=tix, y=bid, legendgroup=str(i), name="bid", line=dict(color="#1f77b4")), row=i, col=1)
                fig7a.add_trace(go.Scatter(x=tix, y=ask, legendgroup=str(i), name="ask", line=dict(color="#d62728")), row=i, col=1)
                fig7a.add_trace(
                    go.Scatter(x=tix, y=theo, legendgroup=str(i), name="theo", line=dict(color="#ff7f0e", width=2)),
                    row=i,
                    col=1,
                )
                fig7a.add_trace(
                    go.Scatter(
                        x=tix,
                        y=mid,
                        legendgroup=str(i),
                        name="mid",
                        mode="markers",
                        marker=dict(size=6, symbol="cross", color="#ffc107"),
                    ),
                    row=i,
                    col=1,
                )
            fig7a.update_layout(
                height=min(2800, 260 * len(panels)),
                showlegend=False,
                title_text="Fig 7a — zoom",
            )
            st.plotly_chart(fig7a, use_container_width=True)

    if "Fig 7b normalized (zoom)" in plot_choice:
        st.caption("Fig 7b — prices minus theoretical (same zoom as 7a).")
        panels = []
        for v in vouchers_sel:
            for d in days_sel:
                zb = _zoom_book(v, d)
                if zb is None:
                    continue
                tix, z = zb
                panels.append((f"{v} day{d}", tix, z))
        if panels:
            fig7b = make_subplots(
                rows=len(panels), cols=1, subplot_titles=[p[0] for p in panels], vertical_spacing=0.05
            )
            for i, (_, tix, z) in enumerate(panels, start=1):
                bid = z["bid_price_1"].astype(float)
                ask = z["ask_price_1"].astype(float)
                mid = z["mid_price"].astype(float)
                theo = z["theo"].astype(float)
                fig7b.add_trace(go.Scatter(x=tix, y=bid - theo, name="bid−theo", line=dict(color="#1f77b4")), row=i, col=1)
                fig7b.add_trace(go.Scatter(x=tix, y=ask - theo, name="ask−theo", line=dict(color="#d62728")), row=i, col=1)
                fig7b.add_trace(
                    go.Scatter(x=tix, y=mid - theo, name="mid−theo", mode="markers", marker=dict(symbol="cross", size=7)),
                    row=i,
                    col=1,
                )
                fig7b.add_hline(y=0, line=dict(color="#ff7f0e", width=2), row=i, col=1)
            fig7b.update_layout(height=min(2800, 240 * len(panels)), showlegend=False, title_text="Fig 7b — normalized zoom")
            st.plotly_chart(fig7b, use_container_width=True)

    if "resdf table sample" in plot_choice:
        st.dataframe(df.sort_values(["day_label", "voucher", "timestamp"]).head(500), use_container_width=True)

    if "Stats summary table" in plot_choice:
        st.dataframe(stats_table(df), use_container_width=True)


if __name__ == "__main__":
    run()
