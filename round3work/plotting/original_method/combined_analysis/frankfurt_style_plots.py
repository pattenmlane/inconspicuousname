"""
Frankfurt Hedgehogs–style figures (Prosperity 3 writeup Fig 6a–6c, 7a–7b, 8)
adapted to Prosperity 4 Round 3 data (VELVETFRUIT_EXTRACT + VEV_* vouchers).

Uses m_t = log(S/K) for smile fit (standard log-moneyness for calls), per timestamp
quadratic v̂(m), same as writeup narrative.

Default: full timestep resolution on historical day=2 for time-series figures;
DTE winds intraday (see plot_iv_smile_round3.t_years_effective). Fig 6a pools IV cloud.
Fig 8 uses underlying mid (VR analog).

Run: python3 round3work/plotting/original_method/combined_analysis/frankfurt_style_plots.py
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_iv_smile_round3 import (
    DATA_DIR,
    STRIKES,
    VOUCHERS,
    bs_call_price,
    dte_from_csv_day,
    fit_smile_poly,
    implied_vol_call,
    load_day_wide,
    t_years_effective,
)

OUT_DIR = Path(__file__).resolve().parent

# Frankfurt-style discrete colors (low strike → high strike, blue → warm)
STRIKE_COLORS: dict[int, str] = {
    4000: "#1f77b4",
    4500: "#d62728",
    5000: "#2ca02c",
    5100: "#9467bd",
    5200: "#ff7f0e",
    5300: "#17becf",
    5400: "#bcbd22",
    5500: "#e377c2",
    6000: "#8c564b",
    6500: "#7f7f7f",
}

# “10k call” analog: focal near-ATM strike for Fig 7a/7b (tune if underlying shifts)
FOCAL_VOUCHER = "VEV_5000"
FOCAL_K = 5000

# Near-ATM band only (cleaner smile like Frankfurt’s tight strike grid around S)
NEAR_MONEY_STRIKES_6A = [5000, 5100, 5200, 5300, 5400, 5500]
# User-requested palette: pink, yellow, blue, orange, purple, green
NEAR_MONEY_COLORS_6A: dict[int, str] = {
    5000: "#e377c2",  # pink
    5100: "#edc949",  # yellow
    5200: "#1f77b4",  # blue
    5300: "#ff7f0e",  # orange
    5400: "#9467bd",  # purple
    5500: "#2ca02c",  # green
}


def apply_frankfurt_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#f4f6f9",
            "axes.edgecolor": "#cccccc",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#ffffff",
            "grid.linewidth": 1.0,
            "grid.alpha": 1.0,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def time_x_axis(n: int, xmax: float = 50_000.0) -> np.ndarray:
    """Map 0..n-1 to 0..xmax like Frankfurt’s ~50k timestep axis."""
    if n <= 1:
        return np.array([0.0])
    return np.linspace(0.0, xmax, n)


def acf_lag1(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 3:
        return float("nan")
    x, y = r[:-1], r[1:]
    if np.std(x) < 1e-14 or np.std(y) < 1e-14:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def build_frankfurt_resdf_full_day(day: int) -> pd.DataFrame:
    """One row per (timestamp, strike): IV, iv_fit from quad in log(S/K), price dev."""
    wide = load_day_wide(day)
    rows: list[dict] = []
    for ts, row in wide.iterrows():
        S = float(row["S"])
        ts_i = int(ts)
        day_i = int(row["day"])
        ty = t_years_effective(day_i, ts_i)
        glist = []
        for v in VOUCHERS:
            if v not in row.index:
                continue
            K = int(v.split("_")[1])
            mid = float(row[v])
            iv = implied_vol_call(mid, S, K, ty, 0.0)
            log_sk = math.log(S / K) if S > 0 and K > 0 else float("nan")
            glist.append(
                {
                    "timestamp": int(ts),
                    "day": int(row["day"]),
                    "dte": int(row["dte"]),
                    "voucher": v,
                    "K": K,
                    "S": S,
                    "mid": mid,
                    "log_S_over_K": log_sk,
                    "iv": iv,
                }
            )
        g = pd.DataFrame(glist)
        if len(g) < 4:
            continue
        fit, res, _ = fit_smile_poly(g, "log_S_over_K", "iv")
        for i in range(len(g)):
            ivf = float(fit[i])
            mid_i = float(g["mid"].iloc[i])
            Ki = float(g["K"].iloc[i])
            theo = (
                bs_call_price(S, Ki, ty, ivf)
                if np.isfinite(ivf) and S > 0 and Ki > 0 and ty > 0
                else float("nan")
            )
            rows.append(
                {
                    "timestamp": ts_i,
                    "day": int(g["day"].iloc[i]),
                    "dte": int(g["dte"].iloc[i]),
                    "t_years": float(ty),
                    "voucher": str(g["voucher"].iloc[i]),
                    "K": int(g["K"].iloc[i]),
                    "S": S,
                    "mid": mid_i,
                    "log_S_over_K": float(g["log_S_over_K"].iloc[i]),
                    "iv": float(g["iv"].iloc[i]),
                    "iv_fit": ivf,
                    "iv_res": float(res[i]),
                    "theoretical_mid": theo,
                    "price_dev": mid_i - theo if np.isfinite(theo) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _figure_6a_base_filtered(resdf: pd.DataFrame) -> pd.DataFrame:
    """Shared filter: valid IV/m, drop tiny extrinsic (Frankfurt outlier policy)."""
    sub = resdf[np.isfinite(resdf["iv"]) & np.isfinite(resdf["log_S_over_K"])].copy()
    sub["intrinsic"] = np.maximum(sub["S"] - sub["K"], 0.0)
    sub = sub[sub["mid"] > sub["intrinsic"] + 0.25]
    return sub


FIGURE_6A_DIR = OUT_DIR / "figure_6a"


def render_figure_6a_full_cloud(resdf: pd.DataFrame, outfile: Path, title: str) -> None:
    """IV vs m_t=log(S/K), all strikes + pooled quadratic fit (full-cloud Figure 6a)."""
    sub = _figure_6a_base_filtered(resdf)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for k in STRIKES:
        g = sub[sub["K"] == k]
        ax.scatter(
            g["log_S_over_K"],
            g["iv"],
            s=2,
            alpha=0.22,
            c=STRIKE_COLORS[k],
            label=f"strike={k}",
            linewidths=0,
            rasterized=True,
        )
    xf = sub["log_S_over_K"].to_numpy(dtype=float)
    yf = sub["iv"].to_numpy(dtype=float)
    m = np.isfinite(xf) & np.isfinite(yf)
    xf, yf = xf[m], yf[m]
    if len(xf) > 100:
        coef = np.polyfit(xf, yf, 2)
        xs = np.linspace(float(np.nanpercentile(xf, 0.5)), float(np.nanpercentile(xf, 99.5)), 300)
        ax.plot(xs, np.polyval(coef, xs), color="black", lw=2.0, label="fitted Parabola", zorder=5)
    ax.set_xlabel(r"$m_t$ (log(S/K))")
    ax.set_ylabel(r"$v_t$ (implied vol)")
    ax.set_title(title)
    ax.legend(loc="upper right", title="variable", framealpha=0.9)
    fig.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(fig)


def figure_6a(resdf: pd.DataFrame) -> None:
    """Default root output (historical day bundled in resdf)."""
    render_figure_6a_full_cloud(
        resdf,
        OUT_DIR / "frankfurt_fig06a_volatility_smile.png",
        "Figure 6a (style): Volatility smile",
    )


def export_figure_6a_per_day_and_combined() -> dict[int, pd.DataFrame]:
    """
    Write round3work/plotting/figure_6a/:
      fig06a_day0_DTE8.png, fig06a_day1_DTE7.png, fig06a_day2_DTE6.png,
      fig06a_combined_days_0_1_2.png
    Returns {day: resdf} for reuse (e.g. day 2 panel for 6b–8).
    """
    FIGURE_6A_DIR.mkdir(parents=True, exist_ok=True)
    parts: dict[int, pd.DataFrame] = {}
    for day in (0, 1, 2):
        dte = dte_from_csv_day(day)
        print(f"figure_6a folder: building day={day} DTE={dte} …")
        resdf = build_frankfurt_resdf_full_day(day)
        parts[day] = resdf
        render_figure_6a_full_cloud(
            resdf,
            FIGURE_6A_DIR / f"fig06a_day{day}_DTE{dte}.png",
            f"Figure 6a (style): Volatility smile — day {day} (DTE={dte})",
        )
    combined = pd.concat(list(parts.values()), ignore_index=True)
    render_figure_6a_full_cloud(
        combined,
        FIGURE_6A_DIR / "fig06a_combined_days_0_1_2.png",
        "Figure 6a (style): Volatility smile — combined days 0–2 (DTE 8, 7, 6)",
    )
    print("Wrote Figure 6a set to", FIGURE_6A_DIR)
    return parts


def figure_6a_near_money(resdf: pd.DataFrame) -> None:
    """
    Near-money SCATTER only (K=5000…5500), but parabola fit on ALL strikes’ pooled cloud
    so v̂(m) matches the full cross-section (Frankfurt-style: one surface, ATM window view).
    Tight axis limits + light styling reduce visual noise vs plotting everything.
    """
    sub_full = _figure_6a_base_filtered(resdf)
    xf_all = sub_full["log_S_over_K"].to_numpy(dtype=float)
    yf_all = sub_full["iv"].to_numpy(dtype=float)
    m_all = np.isfinite(xf_all) & np.isfinite(yf_all)
    xf_all, yf_all = xf_all[m_all], yf_all[m_all]
    coef = np.polyfit(xf_all, yf_all, 2) if len(xf_all) > 100 else None

    sub = sub_full[sub_full["K"].isin(NEAR_MONEY_STRIKES_6A)]
    fig, ax = plt.subplots(figsize=(7.6, 4.35))
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    for k in NEAR_MONEY_STRIKES_6A:
        g = sub[sub["K"] == k]
        ax.scatter(
            g["log_S_over_K"],
            g["iv"],
            s=2,
            alpha=0.34,
            c=NEAR_MONEY_COLORS_6A[k],
            label=f"strike={k}",
            linewidths=0,
            rasterized=True,
        )

    if coef is not None and len(sub) > 0:
        mx = sub["log_S_over_K"].to_numpy(dtype=float)
        mx = mx[np.isfinite(mx)]
        pad = 0.012
        x_lo = float(np.min(mx)) - pad
        x_hi = float(np.max(mx)) + pad
        xs = np.linspace(x_lo, x_hi, 400)
        ax.plot(xs, np.polyval(coef, xs), color="black", lw=2.1, label="fitted Parabola", zorder=6)

    # Tight limits from plotted near-money cloud only (Frankfurt-like framing)
    if len(sub) > 0:
        xv = sub["log_S_over_K"].to_numpy(dtype=float)
        yv = sub["iv"].to_numpy(dtype=float)
        ok = np.isfinite(xv) & np.isfinite(yv)
        xv, yv = xv[ok], yv[ok]
        if len(xv) > 10:
            x_lo, x_hi = np.percentile(xv, [0.2, 99.8])
            y_lo, y_hi = np.percentile(yv, [0.5, 99.5])
            ax.set_xlim(x_lo - 0.015, x_hi + 0.015)
            ax.set_ylim(y_lo - 0.018, y_hi + 0.018)

    ax.set_xlabel(r"$m_t$ (log(S/K))")
    ax.set_ylabel(r"$v_t$ (implied vol)")
    ax.set_title(
        "Figure 6a (style): Volatility smile — near-money strikes (fit uses all strikes)"
    )
    ax.yaxis.grid(True, color="white", linewidth=1.1, alpha=1.0)
    ax.xaxis.grid(False)
    ax.legend(loc="upper right", title="variable", framealpha=0.95, edgecolor="#cccccc")
    fig.tight_layout()
    fig.savefig(
        OUT_DIR / "frankfurt_fig06a_volatility_smile_near_money.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def figure_6b(resdf: pd.DataFrame) -> None:
    """IV − smile IV over time, one line per strike (Frankfurt Fig 6b)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ts_sorted = np.sort(resdf["timestamp"].unique())
    x = time_x_axis(len(ts_sorted))
    ts_to_x = dict(zip(ts_sorted, x))
    for k in STRIKES:
        g = resdf[resdf["K"] == k].set_index("timestamp").reindex(ts_sorted)
        y = g["iv_res"].to_numpy(dtype=float)
        ax.plot(x, y, color=STRIKE_COLORS[k], lw=0.7, label=f"strike={k}", alpha=0.9)
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_xlim(0, 50_000)
    ax.set_xlabel("timestamp")
    ax.set_ylabel(r"Option_IV − VolSmile_IV ($v_t - \hat v_t$)")
    ax.set_title("Figure 6b (style): IV deviations over time")
    ax.legend(loc="upper right", title="variable", ncol=1, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "frankfurt_fig06b_iv_deviations_time.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def figure_6c(resdf: pd.DataFrame) -> None:
    """Mid − BS(theo using smile IV) over time (Frankfurt Fig 6c)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ts_sorted = np.sort(resdf["timestamp"].unique())
    x = time_x_axis(len(ts_sorted))
    for k in STRIKES:
        g = resdf[resdf["K"] == k].set_index("timestamp").reindex(ts_sorted)
        y = g["price_dev"].to_numpy(dtype=float)
        ax.plot(x, y, color=STRIKE_COLORS[k], lw=0.7, label=f"strike={k}", alpha=0.9)
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_xlim(0, 50_000)
    ax.set_xlabel("timestamp")
    ax.set_ylabel("Option_Price − BS_theo(VolSmile_IV)")
    ax.set_title("Figure 6c (style): Price deviations over time")
    ax.legend(loc="upper right", title="variable", ncol=1, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "frankfurt_fig06c_price_deviations_time.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def load_book_product(day: int, product: str) -> pd.DataFrame:
    path = DATA_DIR / f"prices_round_3_day_{day}.csv"
    df = pd.read_csv(path, sep=";")
    return df[df["product"] == product].sort_values("timestamp").reset_index(drop=True)


def figure_7a_7b(resdf: pd.DataFrame, day: int, zoom_len: int = 51, zoom_start_idx: int = 4000) -> None:
    """Bid/ask/theo/mid zoom; normalized version (Frankfurt 7a/7b style)."""
    book = load_book_product(day, FOCAL_VOUCHER)
    theo_map = (
        resdf[resdf["voucher"] == FOCAL_VOUCHER][["timestamp", "theoretical_mid"]]
        .drop_duplicates("timestamp")
        .set_index("timestamp")["theoretical_mid"]
    )
    book = book.copy()
    book["theo"] = book["timestamp"].map(theo_map)
    z = book.iloc[zoom_start_idx : zoom_start_idx + zoom_len].copy()
    n = len(z)
    if n < 5:
        return
    tix = np.arange(n)
    bid = z["bid_price_1"].to_numpy(dtype=float)
    ask = z["ask_price_1"].to_numpy(dtype=float)
    mid = z["mid_price"].to_numpy(dtype=float)
    theo = z["theo"].to_numpy(dtype=float)
    bv = z["bid_volume_1"].fillna(1).to_numpy(dtype=float)
    av = z["ask_volume_1"].fillna(1).to_numpy(dtype=float)
    bsz = 15 + 80 * (bv / (np.nanmax(bv) + 1e-6))
    asz = 15 + 80 * (av / (np.nanmax(av) + 1e-6))

    # --- 7a ---
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(tix, bid, color="#1f77b4", lw=0.8, alpha=0.9, zorder=1)
    ax.plot(tix, ask, color="#d62728", lw=0.8, alpha=0.9, zorder=1)
    ax.scatter(tix, bid, s=bsz, color="#1f77b4", edgecolors="none", zorder=2, alpha=0.85)
    ax.scatter(tix, ask, s=asz, color="#d62728", edgecolors="none", zorder=2, alpha=0.85)
    ax.plot(tix, theo, color="#ff7f0e", lw=1.8, label="theoretical (BS @ smile IV)", zorder=3)
    ax.scatter(tix, mid, marker="+", s=120, c="#ffc107", linewidths=1.2, zorder=4, label="mid")
    ax.set_xlabel("time index (zoom window)")
    ax.set_ylabel("price")
    ax.set_title(f"Figure 7a (style): {FOCAL_VOUCHER} bid/ask vs theoretical")
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "frankfurt_fig07a_focal_call_fluctuations.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # --- 7b normalized ---
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.axhline(0, color="#ff7f0e", lw=2.2, label="theoretical (=0)")
    ax.plot(tix, bid - theo, color="#1f77b4", lw=0.8, alpha=0.9)
    ax.plot(tix, ask - theo, color="#d62728", lw=0.8, alpha=0.9)
    ax.scatter(tix, bid - theo, s=bsz, color="#1f77b4", edgecolors="none", alpha=0.85)
    ax.scatter(tix, ask - theo, s=asz, color="#d62728", edgecolors="none", alpha=0.85)
    ax.scatter(tix, mid - theo, marker="+", s=120, c="#ffc107", linewidths=1.2, label="mid − theo")
    ax.set_xlabel("time index (zoom window)")
    ax.set_ylabel("price − theoretical")
    ax.set_title(f"Figure 7b (style): {FOCAL_VOUCHER} normalized")
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "frankfurt_fig07b_focal_call_fluctuations_normalized.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def figure_8_underlying_acf(day: int, n_random: int = 120) -> None:
    """Rolling lag-1 ACF on trailing window w vs random return paths (Fig 8 style)."""
    wide = load_day_wide(day)
    S = wide["S"].to_numpy(dtype=float)
    r = np.diff(np.log(S))
    r = r[np.isfinite(r)]
    N = len(r)
    std = float(np.nanstd(r)) or 1.0
    ws = np.arange(5, 101, dtype=int)
    y_vr = np.full(len(ws), np.nan)
    for i, w in enumerate(ws):
        y_vr[i] = acf_lag1(r[N - w : N])

    rng = np.random.default_rng(42)
    y_rand = np.full((n_random, len(ws)), np.nan)
    for t in range(n_random):
        noise = rng.standard_normal(N) * std
        for i, w in enumerate(ws):
            y_rand[t, i] = acf_lag1(noise[N - w : N])

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for t in range(n_random):
        ax.plot(ws, y_rand[t], color="black", lw=0.35, alpha=0.18)
    ax.plot(ws, y_vr, color="#d62728", lw=2.0, label="VELVETFRUIT_EXTRACT")
    ax.set_xlabel("rolling window")
    ax.set_ylabel("Autocorrelation (lag 1)")
    ax.set_title("Figure 8 (style): underlying return ACF vs random")
    ax.legend(loc="lower left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "frankfurt_fig08_underlying_rolling_acf.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    apply_frankfurt_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    day = 2
    by_day = export_figure_6a_per_day_and_combined()
    resdf = by_day[day]
    resdf.to_csv(OUT_DIR / "frankfurt_resdf_full_day.csv", index=False)
    print("rows", len(resdf), f"(day {day}) — plotting 6a root copy, near-money, 6b–8")
    figure_6a(resdf)
    figure_6a_near_money(resdf)
    figure_6b(resdf)
    figure_6c(resdf)
    figure_7a_7b(resdf, day=day)
    figure_8_underlying_acf(day=day)
    print("Wrote Frankfurt-style figures to", OUT_DIR)


if __name__ == "__main__":
    main()
