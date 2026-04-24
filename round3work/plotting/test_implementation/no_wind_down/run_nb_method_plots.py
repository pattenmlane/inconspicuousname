"""
Regenerate the plotting-folder figure set using round3.ipynb methodology (see nb_method_core).

Output: this folder (`test_implementation/no_wind_down/`). T = DTE_open/365 only (no wind-down).

Run:  python3 round3work/plotting/test_implementation/no_wind_down/run_nb_method_plots.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_PLOT_DIR = Path(__file__).resolve().parent.parent.parent
_ORIG = _PLOT_DIR / "original_method" / "no_wind_down" / "combined_analysis"
for _p in (_ORIG, _PLOT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import frankfurt_style_plots as fs
from nb_method_core import (
    STRIKES,
    VOUCHERS,
    black_scholes,
    build_full_resdf_nb_day,
    build_ivdf_nb_all_days,
    build_resdf_nb_from_ivdf,
    dte_from_csv_day,
    load_book_product,
    load_day_wide,
)

OUT_DIR = Path(__file__).resolve().parent
FIG6_DIR = OUT_DIR / "figure_6a"
FOCAL = "VEV_5000"


def _figure_6a_base_filtered(resdf: pd.DataFrame) -> pd.DataFrame:
    sub = resdf[np.isfinite(resdf["iv"]) & np.isfinite(resdf["m_nb"])].copy()
    sub["intrinsic"] = np.maximum(sub["S"] - sub["K"], 0.0)
    sub = sub[sub["mid"] > sub["intrinsic"] + 0.25]
    return sub


def render_fig6a_full(resdf: pd.DataFrame, outfile: Path, title: str) -> None:
    sub = _figure_6a_base_filtered(resdf)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for k in STRIKES:
        g = sub[sub["K"] == k]
        ax.scatter(
            g["m_nb"],
            g["iv"],
            s=2,
            alpha=0.22,
            c=fs.STRIKE_COLORS[k],
            label=f"strike={k}",
            linewidths=0,
            rasterized=True,
        )
    xf = sub["m_nb"].to_numpy(dtype=float)
    yf = sub["iv"].to_numpy(dtype=float)
    m = np.isfinite(xf) & np.isfinite(yf)
    xf, yf = xf[m], yf[m]
    if len(xf) > 100:
        coef = np.polyfit(xf, yf, 2)
        xs = np.linspace(float(np.nanpercentile(xf, 0.5)), float(np.nanpercentile(xf, 99.5)), 300)
        ax.plot(xs, np.polyval(coef, xs), color="black", lw=2.0, label="fitted Parabola", zorder=5)
    ax.set_xlabel(r"$m=\log(K/S)/\sqrt{T}$ (notebook)")
    ax.set_ylabel(r"$v$ (implied vol, bisection [0.01,1])")
    ax.set_title(title)
    ax.legend(loc="upper right", title="variable", framealpha=0.9)
    fig.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_fig6a_near_money(resdf: pd.DataFrame, outfile: Path) -> None:
    sub_full = _figure_6a_base_filtered(resdf)
    xf_all = sub_full["m_nb"].to_numpy(dtype=float)
    yf_all = sub_full["iv"].to_numpy(dtype=float)
    m_all = np.isfinite(xf_all) & np.isfinite(yf_all)
    xf_all, yf_all = xf_all[m_all], yf_all[m_all]
    coef = np.polyfit(xf_all, yf_all, 2) if len(xf_all) > 100 else None
    sub = sub_full[sub_full["K"].isin(fs.NEAR_MONEY_STRIKES_6A)]
    fig, ax = plt.subplots(figsize=(7.6, 4.35))
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for k in fs.NEAR_MONEY_STRIKES_6A:
        g = sub[sub["K"] == k]
        ax.scatter(
            g["m_nb"],
            g["iv"],
            s=2,
            alpha=0.34,
            c=fs.NEAR_MONEY_COLORS_6A[k],
            label=f"strike={k}",
            linewidths=0,
            rasterized=True,
        )
    if coef is not None and len(sub) > 0:
        mx = sub["m_nb"].to_numpy(dtype=float)
        mx = mx[np.isfinite(mx)]
        pad = 0.05
        x_lo, x_hi = float(np.min(mx)) - pad, float(np.max(mx)) + pad
        xs = np.linspace(x_lo, x_hi, 400)
        ax.plot(xs, np.polyval(coef, xs), color="black", lw=2.1, label="fitted Parabola", zorder=6)
    if len(sub) > 0:
        xv = sub["m_nb"].to_numpy(dtype=float)
        yv = sub["iv"].to_numpy(dtype=float)
        ok = np.isfinite(xv) & np.isfinite(yv)
        xv, yv = xv[ok], yv[ok]
        if len(xv) > 10:
            x_lo, x_hi = np.percentile(xv, [0.2, 99.8])
            y_lo, y_hi = np.percentile(yv, [0.5, 99.5])
            ax.set_xlim(x_lo - 0.02, x_hi + 0.02)
            ax.set_ylim(y_lo - 0.018, y_hi + 0.018)
    ax.set_xlabel(r"$m=\log(K/S)/\sqrt{T}$")
    ax.set_ylabel(r"$v$ (IV)")
    ax.set_title("Fig 6a near-money — global polyfit on m, scatter 5000–5500 (notebook method)")
    ax.yaxis.grid(True, color="white", linewidth=1.1, alpha=1.0)
    ax.xaxis.grid(False)
    ax.legend(loc="upper right", title="variable", framealpha=0.95, edgecolor="#cccccc")
    fig.tight_layout()
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def export_figure_6a_folder() -> dict[int, pd.DataFrame]:
    FIG6_DIR.mkdir(parents=True, exist_ok=True)
    parts = {}
    for day in (0, 1, 2):
        dte = dte_from_csv_day(day)
        print(f"  figure_6a (nb): day={day} DTE_open={dte} …")
        r = build_full_resdf_nb_day(day)
        parts[day] = r
        render_fig6a_full(
            r,
            FIG6_DIR / f"fig06a_day{day}_DTE{dte}.png",
            f"Figure 6a (notebook method): day {day} (DTE open={dte})",
        )
    comb = pd.concat(list(parts.values()), ignore_index=True)
    render_fig6a_full(
        comb,
        FIG6_DIR / "fig06a_combined_days_0_1_2.png",
        "Figure 6a (notebook method): combined days 0–2",
    )
    return parts


def plot_iv_style(ivdf: pd.DataFrame, resdf: pd.DataFrame) -> None:
    snap_times = [0, 500_000, 999_900]
    strike_to_color = {k: plt.cm.tab10(i / 9.0) for i, k in enumerate(STRIKES)}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for ax, day in zip(axes, (0, 1, 2)):
        sub = ivdf[ivdf["day"] == day]
        for t in snap_times:
            g = sub[sub["timestamp"] == t].sort_values("K")
            ax.plot(g["m_nb"], g["iv"], "o-", alpha=0.85, markersize=5, label=f"t={t}")
        ax.set_title(f"day={day} (DTE open={dte_from_csv_day(day)}, flat T)")
        ax.set_xlabel(r"$m=\log(K/S)/\sqrt{T}$")
        ax.set_ylabel("IV")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("IV smile snapshots (notebook method)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "iv_smile_snapshots.png", dpi=160)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for ax, day in zip(axes, (0, 1, 2)):
        sub = ivdf[ivdf["day"] == day]
        for k in STRIKES:
            g = sub[sub["K"] == k]
            ax.scatter(
                g["m_nb"],
                g["iv"],
                s=4,
                alpha=0.35,
                c=[strike_to_color[k]] * len(g),
                label=f"{k}" if day == 0 else None,
            )
        ax.set_title(f"day={day} (subsampled step=20)")
        ax.set_xlabel(r"$m=\log(K/S)/\sqrt{T}$")
        ax.set_ylabel("IV")
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=7, ncol=2)
    fig.suptitle("IV vs m (notebook moneyness)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.92, 0.96])
    fig.savefig(OUT_DIR / "iv_vs_logmoneyness_scatter.png", dpi=160)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for day in (0, 1, 2):
        sub = ivdf[ivdf["day"] == day]
        m = sub.groupby("K")["iv"].median()
        ax.plot(m.index, m.values, "o-", label=f"day {day} DTE open {dte_from_csv_day(day)}")
    ax.set_xlabel("strike K")
    ax.set_ylabel("median IV")
    ax.set_title("Median IV by strike (notebook IV)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "median_iv_by_strike.png", dpi=160)
    plt.close()

    summary = resdf.groupby(["day", "voucher"])["iv_res"].agg(
        mean="mean",
        std="std",
        mean_abs_res=lambda s: float(np.nanmean(np.abs(s))),
    )
    summary.to_csv(OUT_DIR / "iv_smile_residual_summary.csv")

    fig, ax = plt.subplots(figsize=(9, 4))
    for day in (0, 1, 2):
        r = resdf[resdf["day"] == day]
        ts_list = sorted(r["timestamp"].unique())[::10]
        rr = r[r["timestamp"].isin(ts_list)]
        ax.scatter(rr["m_nb"], rr["iv_res"], s=6, alpha=0.25, label=f"day {day}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(r"$m=\log(K/S)/\sqrt{T}$")
    ax.set_ylabel("IV − quadratic fit(m)")
    ax.set_title("Detrended IV (notebook m)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "iv_residuals_detrended.png", dpi=160)
    plt.close()

    ivdf.sample(min(50000, len(ivdf)), random_state=0).to_csv(
        OUT_DIR / "iv_panel_sample.csv", index=False
    )


def lag1_autocorr(series: pd.Series) -> float:
    s = series.dropna()
    if len(s) < 3:
        return float("nan")
    a = s.to_numpy(dtype=float)
    x, y = a[:-1], a[1:]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def plot_price_deviation_analysis(resdf: pd.DataFrame) -> None:
    T_arr = resdf["T_years"].to_numpy(dtype=float)
    iv_fit = resdf["iv_fit"].to_numpy(dtype=float)
    S = resdf["S"].to_numpy(dtype=float)
    K = resdf["K"].to_numpy(dtype=float)
    mid = resdf["mid"].to_numpy(dtype=float)
    n = len(resdf)
    theo = np.full(n, np.nan, dtype=float)
    for i in range(n):
        sig = iv_fit[i]
        if not np.isfinite(sig) or S[i] <= 0 or K[i] <= 0 or T_arr[i] <= 0:
            continue
        theo[i] = black_scholes(float(S[i]), float(K[i]), float(T_arr[i]), 0.0, float(sig))
    resdf = resdf.assign(theoretical_mid=theo, price_deviation=mid - theo)

    rows = []
    for (day, v), g in resdf.groupby(["day", "voucher"]):
        g = g.sort_values("timestamp")
        rows.append(
            {
                "day": day,
                "voucher": v,
                "lag1_autocorr_price_dev": lag1_autocorr(g["price_deviation"]),
                "mean_abs_price_dev": float(np.nanmean(np.abs(g["price_deviation"]))),
            }
        )
    pd.DataFrame(rows).to_csv(OUT_DIR / "price_deviation_lag1_autocorr.csv", index=False)

    ac_tbl = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, day in zip(axes, (0, 1, 2)):
        sub = ac_tbl[ac_tbl["day"] == day].set_index("voucher").reindex([f"VEV_{k}" for k in STRIKES])
        ys = sub["lag1_autocorr_price_dev"].to_numpy(dtype=float)
        ax.bar(range(len(STRIKES)), ys, tick_label=[str(k) for k in STRIKES])
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"day {day} (notebook method)")
        ax.set_xlabel("strike")
        ax.set_ylabel("lag-1 autocorr(dev)")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Price deviation lag-1 autocorr (notebook BS + smile IV)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "price_deviation_lag1_autocorr_bars.png", dpi=160)
    plt.close()

    for target in ("VEV_5000", "VEV_5100"):
        fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
        for ax, day in zip(axes, (0, 1, 2)):
            g = resdf[(resdf["day"] == day) & (resdf["voucher"] == target)].sort_values("timestamp")
            ax.plot(g["timestamp"], g["price_deviation"], lw=0.8, alpha=0.9)
            ax.set_ylabel("dev")
            ax.set_title(f"{target} day={day}")
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("timestamp")
        fig.suptitle(f"Price deviation vs time — {target} (notebook)", fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"price_deviation_timeseries_{target}.png", dpi=160)
        plt.close()


def plot_frankfurt_style_ts(resdf: pd.DataFrame, day: int = 2) -> None:
    ts_sorted = np.sort(resdf["timestamp"].unique())
    x = fs.time_x_axis(len(ts_sorted))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for k in STRIKES:
        g = resdf[resdf["K"] == k].set_index("timestamp").reindex(ts_sorted)
        y = g["iv_res"].to_numpy(dtype=float)
        ax.plot(x, y, color=fs.STRIKE_COLORS[k], lw=0.7, label=f"strike={k}", alpha=0.9)
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_xlim(0, 50_000)
    ax.set_xlabel("timestamp")
    ax.set_ylabel(r"IV − smile IV ($m=\log(K/S)/\sqrt{T}$ quad)")
    ax.set_title("Fig 6b (notebook method): IV deviations over time")
    ax.legend(loc="upper right", title="variable", ncol=1, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "frankfurt_fig06b_iv_deviations_time.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for k in STRIKES:
        g = resdf[resdf["K"] == k].set_index("timestamp").reindex(ts_sorted)
        y = g["price_dev"].to_numpy(dtype=float)
        ax.plot(x, y, color=fs.STRIKE_COLORS[k], lw=0.7, label=f"strike={k}", alpha=0.9)
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_xlim(0, 50_000)
    ax.set_xlabel("timestamp")
    ax.set_ylabel("mid − BS(smile IV)")
    ax.set_title("Fig 6c (notebook method): price deviations over time")
    ax.legend(loc="upper right", title="variable", ncol=1, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "frankfurt_fig06c_price_deviations_time.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    book = load_book_product(day, FOCAL)
    theo_map = (
        resdf[resdf["voucher"] == FOCAL][["timestamp", "theoretical_mid"]]
        .drop_duplicates("timestamp")
        .set_index("timestamp")["theoretical_mid"]
    )
    book = book.copy()
    book["theo"] = book["timestamp"].map(theo_map)
    z = book.iloc[4000 : 4000 + 51].copy()
    n = len(z)
    if n >= 5:
        tix = np.arange(n)
        bid = z["bid_price_1"].to_numpy(float)
        ask = z["ask_price_1"].to_numpy(float)
        mid = z["mid_price"].to_numpy(float)
        theo = z["theo"].to_numpy(float)
        bv = z["bid_volume_1"].fillna(1).to_numpy(float)
        av = z["ask_volume_1"].fillna(1).to_numpy(float)
        bsz = 15 + 80 * (bv / (np.nanmax(bv) + 1e-6))
        asz = 15 + 80 * (av / (np.nanmax(av) + 1e-6))
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.plot(tix, bid, color="#1f77b4", lw=0.8, alpha=0.9, zorder=1)
        ax.plot(tix, ask, color="#d62728", lw=0.8, alpha=0.9, zorder=1)
        ax.scatter(tix, bid, s=bsz, color="#1f77b4", edgecolors="none", zorder=2, alpha=0.85)
        ax.scatter(tix, ask, s=asz, color="#d62728", edgecolors="none", zorder=2, alpha=0.85)
        ax.plot(tix, theo, color="#ff7f0e", lw=1.8, zorder=3, label="theoretical")
        ax.scatter(tix, mid, marker="+", s=120, c="#ffc107", linewidths=1.2, zorder=4, label="mid")
        ax.set_xlabel("time index (zoom)")
        ax.set_ylabel("price")
        ax.set_title(f"Fig 7a (notebook): {FOCAL}")
        ax.legend(loc="best", framealpha=0.9)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "frankfurt_fig07a_focal_call_fluctuations.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.axhline(0, color="#ff7f0e", lw=2.2, label="theoretical (=0)")
        ax.plot(tix, bid - theo, color="#1f77b4", lw=0.8, alpha=0.9)
        ax.plot(tix, ask - theo, color="#d62728", lw=0.8, alpha=0.9)
        ax.scatter(tix, bid - theo, s=bsz, color="#1f77b4", edgecolors="none", alpha=0.85)
        ax.scatter(tix, ask - theo, s=asz, color="#d62728", edgecolors="none", alpha=0.85)
        ax.scatter(tix, mid - theo, marker="+", s=120, c="#ffc107", linewidths=1.2, label="mid − theo")
        ax.set_xlabel("time index (zoom)")
        ax.set_ylabel("price − theoretical")
        ax.set_title(f"Fig 7b (notebook): {FOCAL} normalized")
        ax.legend(loc="best", framealpha=0.9)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "frankfurt_fig07b_focal_call_fluctuations_normalized.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def plot_fig8(day: int = 2) -> None:
    wide = load_day_wide(day)
    S = wide["S"].to_numpy(dtype=float)
    r = np.diff(np.log(S))
    r = r[np.isfinite(r)]
    N = len(r)
    std = float(np.nanstd(r)) or 1.0
    ws = np.arange(5, 101, dtype=int)
    rng = np.random.default_rng(42)
    n_random = 120

    def acf1(a: np.ndarray) -> float:
        a = np.asarray(a, dtype=float)
        a = a[np.isfinite(a)]
        if len(a) < 3:
            return float("nan")
        x, y = a[:-1], a[1:]
        if np.std(x) < 1e-14 or np.std(y) < 1e-14:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    y_vr = np.full(len(ws), np.nan)
    for i, w in enumerate(ws):
        y_vr[i] = acf1(r[N - w : N])
    y_rand = np.full((n_random, len(ws)), np.nan)
    for t in range(n_random):
        noise = rng.standard_normal(N) * std
        for i, w in enumerate(ws):
            y_rand[t, i] = acf1(noise[N - w : N])
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for t in range(n_random):
        ax.plot(ws, y_rand[t], color="black", lw=0.35, alpha=0.18)
    ax.plot(ws, y_vr, color="#d62728", lw=2.0, label="VELVETFRUIT_EXTRACT")
    ax.set_xlabel("rolling window")
    ax.set_ylabel("Autocorrelation (lag 1)")
    ax.set_title("Fig 8: underlying return ACF vs random")
    ax.legend(loc="lower left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "frankfurt_fig08_underlying_rolling_acf.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fs.apply_frankfurt_style()
    print("Notebook-method plots →", OUT_DIR)

    print("1) Subsampled IV panel (step=20) …")
    ivdf = build_ivdf_nb_all_days(step=20)
    resdf_light = build_resdf_nb_from_ivdf(ivdf)
    plot_iv_style(ivdf, resdf_light)
    plot_price_deviation_analysis(resdf_light.copy())

    print("2) Full-day resdf per figure_6a + Frankfurt 6a/6b/6c/7 …")
    by_day = export_figure_6a_folder()
    resdf2 = by_day[2]
    resdf2.to_csv(OUT_DIR / "frankfurt_resdf_full_day.csv", index=False)
    render_fig6a_full(
        resdf2,
        OUT_DIR / "frankfurt_fig06a_volatility_smile.png",
        "Figure 6a (notebook method): day 2 full cloud",
    )
    render_fig6a_near_money(resdf2, OUT_DIR / "frankfurt_fig06a_volatility_smile_near_money.png")
    plot_frankfurt_style_ts(resdf2, day=2)
    plot_fig8(day=2)

    print("Done.")


if __name__ == "__main__":
    main()
