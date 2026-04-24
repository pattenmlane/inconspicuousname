"""
Per-voucher diagnostics under pervoucher_analysis/<VOUCHER>/{day0,day1,day2,combined}/.

Uses the same model as combined_analysis (winding DTE, BS, quadratic smile in log(S/K)).

Run from repo:
  python3 round3work/plotting/original_method/pervoucher_analysis/build_pervoucher.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent
_COMBINED = _ROOT.parent / "combined_analysis"
if str(_COMBINED) not in sys.path:
    sys.path.insert(0, str(_COMBINED))

import frankfurt_style_plots as fs
from frankfurt_style_plots import (
    VOUCHERS,
    acf_lag1,
    apply_frankfurt_style,
    build_frankfurt_resdf_full_day,
    load_book_product,
    load_day_wide,
    time_x_axis,
)
from plot_iv_smile_round3 import dte_from_csv_day

OUT_PV = _ROOT
ZOOM_START = 4000
ZOOM_LEN = 51
X_SEGMENT = 55_000.0  # gap between days on combined x-axis


def lag1_autocorr(series: pd.Series) -> float:
    s = series.dropna()
    if len(s) < 3:
        return float("nan")
    a = s.to_numpy(dtype=float)
    x, y = a[:-1], a[1:]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def plot_fig8_days_to_common() -> None:
    common = OUT_PV / "_common"
    common.mkdir(parents=True, exist_ok=True)
    for day in (0, 1, 2):
        wide = load_day_wide(day)
        S = wide["S"].to_numpy(dtype=float)
        r = np.diff(np.log(S))
        r = r[np.isfinite(r)]
        N = len(r)
        std = float(np.nanstd(r)) or 1.0
        ws = np.arange(5, 101, dtype=int)
        rng = np.random.default_rng(42)
        n_random = 120
        y_vr = np.full(len(ws), np.nan)
        for i, w in enumerate(ws):
            y_vr[i] = acf_lag1(r[N - w : N])
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
        ax.set_title(f"Figure 8 (style): underlying return ACF vs random — day {day}")
        ax.legend(loc="lower left", framealpha=0.9)
        fig.tight_layout()
        fig.savefig(common / f"underlying_fig08_day{day}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def plot_7a_7b(
    resdf: pd.DataFrame,
    day: int,
    voucher: str,
    out_dir: Path,
    zoom_start: int = ZOOM_START,
    zoom_len: int = ZOOM_LEN,
) -> None:
    book = load_book_product(day, voucher)
    theo_map = (
        resdf[resdf["voucher"] == voucher][["timestamp", "theoretical_mid"]]
        .drop_duplicates("timestamp")
        .set_index("timestamp")["theoretical_mid"]
    )
    book = book.copy()
    book["theo"] = book["timestamp"].map(theo_map)
    z = book.iloc[zoom_start : zoom_start + zoom_len].copy()
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

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(tix, bid, color="#1f77b4", lw=0.8, alpha=0.9, zorder=1)
    ax.plot(tix, ask, color="#d62728", lw=0.8, alpha=0.9, zorder=1)
    ax.scatter(tix, bid, s=bsz, color="#1f77b4", edgecolors="none", zorder=2, alpha=0.85)
    ax.scatter(tix, ask, s=asz, color="#d62728", edgecolors="none", zorder=2, alpha=0.85)
    ax.plot(tix, theo, color="#ff7f0e", lw=1.8, label="theoretical (BS @ smile IV)", zorder=3)
    ax.scatter(tix, mid, marker="+", s=120, c="#ffc107", linewidths=1.2, zorder=4, label="mid")
    ax.set_xlabel("time index (zoom window)")
    ax.set_ylabel("price")
    ax.set_title(f"Figure 7a (style): {voucher} bid/ask vs theoretical — day {day}")
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_dir / "fig07a_bid_ask_theo_mid.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.axhline(0, color="#ff7f0e", lw=2.2, label="theoretical (=0)")
    ax.plot(tix, bid - theo, color="#1f77b4", lw=0.8, alpha=0.9)
    ax.plot(tix, ask - theo, color="#d62728", lw=0.8, alpha=0.9)
    ax.scatter(tix, bid - theo, s=bsz, color="#1f77b4", edgecolors="none", alpha=0.85)
    ax.scatter(tix, ask - theo, s=asz, color="#d62728", edgecolors="none", alpha=0.85)
    ax.scatter(tix, mid - theo, marker="+", s=120, c="#ffc107", linewidths=1.2, label="mid − theo")
    ax.set_xlabel("time index (zoom window)")
    ax.set_ylabel("price − theoretical")
    ax.set_title(f"Figure 7b (style): {voucher} normalized — day {day}")
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_dir / "fig07b_normalized.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_day_bundle(
    resdf_full_day: pd.DataFrame,
    day: int,
    voucher: str,
    out_dir: Path,
) -> dict[str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    g = resdf_full_day[resdf_full_day["voucher"] == voucher].copy()
    if len(g) < 10:
        return {}
    g = g.sort_values("timestamp")
    ts_sorted = np.sort(g["timestamp"].unique())
    x = time_x_axis(len(ts_sorted))
    gg = g.set_index("timestamp").reindex(ts_sorted)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(x, gg["iv_res"].to_numpy(dtype=float), color=fs.STRIKE_COLORS[int(voucher.split("_")[1])], lw=0.75)
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_xlim(0, 50_000)
    ax.set_xlabel("session time index (Frankfurt-style)")
    ax.set_ylabel(r"$v - \hat v$ (IV minus smile fit)")
    ax.set_title(f"Fig 6b analog — {voucher} — day {day} (DTE open {dte_from_csv_day(day)})")
    fig.tight_layout()
    fig.savefig(out_dir / "fig06b_iv_residual_vs_time.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(x, gg["price_dev"].to_numpy(dtype=float), color=fs.STRIKE_COLORS[int(voucher.split("_")[1])], lw=0.75)
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_xlim(0, 50_000)
    ax.set_xlabel("session time index")
    ax.set_ylabel("mid − BS(smile IV)")
    ax.set_title(f"Fig 6c analog — {voucher} — day {day}")
    fig.tight_layout()
    fig.savefig(out_dir / "fig06c_price_dev_vs_time.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(x, gg["iv"].to_numpy(dtype=float), color="#2ca02c", lw=0.75, label="market IV")
    ax.plot(x, gg["iv_fit"].to_numpy(dtype=float), color="#9467bd", lw=0.75, alpha=0.9, label="smile IV̂")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 50_000)
    ax.set_xlabel("session time index")
    ax.set_ylabel("implied vol")
    ax.set_title(f"Market IV vs smile-fitted IV — {voucher} — day {day}")
    fig.tight_layout()
    fig.savefig(out_dir / "iv_vs_ivfit_time.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    log_ks = np.log(gg["K"].to_numpy(dtype=float) / gg["S"].to_numpy(dtype=float))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(x, log_ks, color="#17becf", lw=0.65)
    ax.set_xlim(0, 50_000)
    ax.set_xlabel("session time index")
    ax.set_ylabel("log(K/S)")
    ax.set_title(f"Moneyness path — {voucher} — day {day}")
    fig.tight_layout()
    fig.savefig(out_dir / "log_K_over_S_vs_time.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    book = load_book_product(day, voucher)
    book = book.set_index("timestamp").reindex(ts_sorted)
    spread = (book["ask_price_1"] - book["bid_price_1"]).to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(x, spread, color="#8c564b", lw=0.65)
    ax.set_xlim(0, 50_000)
    ax.set_xlabel("session time index")
    ax.set_ylabel("ask − bid")
    ax.set_title(f"Quoted spread — {voucher} — day {day}")
    fig.tight_layout()
    fig.savefig(out_dir / "spread_vs_time.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    plot_7a_7b(resdf_full_day, day, voucher, out_dir)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    axes[0].hist(gg["price_dev"].dropna(), bins=40, color="#1f77b4", alpha=0.85, edgecolor="white")
    axes[0].set_title("price deviation")
    axes[0].set_xlabel("mid − theo")
    axes[1].hist(gg["iv_res"].dropna(), bins=40, color="#ff7f0e", alpha=0.85, edgecolor="white")
    axes[1].set_title("IV residual")
    axes[1].set_xlabel(r"$v-\hat v$")
    fig.suptitle(f"Distributions — {voucher} — day {day}")
    fig.tight_layout()
    fig.savefig(out_dir / "hist_price_dev_and_iv_res.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    g.to_csv(out_dir / "resdf_slice.csv", index=False)
    stats = {
        "lag1_price_dev": lag1_autocorr(gg["price_dev"]),
        "lag1_iv_res": lag1_autocorr(gg["iv_res"]),
        "mean_abs_price_dev": float(np.nanmean(np.abs(gg["price_dev"]))),
        "mean_abs_iv_res": float(np.nanmean(np.abs(gg["iv_res"]))),
        "mean_iv_res": float(np.nanmean(gg["iv_res"])),
        "median_spread": float(np.nanmedian(spread)),
        "median_log_KS": float(np.nanmedian(log_ks)),
    }
    pd.DataFrame([stats]).to_csv(out_dir / "stats_summary.csv", index=False)
    return stats


def plot_combined_timeline(
    parts: list[pd.DataFrame],
    voucher: str,
    out_dir: Path,
    ycol: str,
    fname: str,
    ylabel: str,
    title: str,
) -> None:
    xs, ys = [], []
    for day_i, df in enumerate(parts):
        g = df[df["voucher"] == voucher].sort_values("timestamp")
        if len(g) < 5:
            continue
        ts = np.sort(g["timestamp"].unique())
        xloc = time_x_axis(len(ts))
        gg = g.set_index("timestamp").reindex(ts)
        off = day_i * X_SEGMENT
        xs.append(xloc + off)
        ys.append(gg[ycol].to_numpy(dtype=float))
    if not xs:
        return
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    ax.plot(x, y, color=fs.STRIKE_COLORS[int(voucher.split("_")[1])], lw=0.65)
    ax.axhline(0, color="#333333", lw=0.8)
    for b in range(1, len(parts)):
        ax.axvline(b * X_SEGMENT, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("concatenated session index (day0 | day1 | day2)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_findings(
    voucher_path: Path,
    voucher: str,
    K: int,
    by_day: dict[int, dict[str, float]],
    combined_rows: list[dict],
) -> None:
    lines = [
        "=" * 78,
        f"PER-VOUCHER FINDINGS — {voucher} (K={K})",
        "Round 3 historical days 0–1–2, original pipeline (combined_analysis).",
        "=" * 78,
        "",
        "OVERALL DIRECTION (aligned with suggested_approach.txt + questions_and_answers.txt)",
        "-" * 78,
        "We treat the quadratic smile in log(S/K) as a deliberate cross-sectional benchmark,",
        "not truth. Edge lives in price space after spreads: mid − BS(S,K,T, σ̂_surface).",
        "Diagnostics below tell you whether THIS strike behaves like a clean relative-value",
        "name (good IV-scalping candidate) or a sticky/persistent one (caution for naive",
        "mean-reversion), and whether quoted spread eats the typical deviation.",
        "",
        "KEY METRICS BY DAY",
        "-" * 78,
    ]
    for day in sorted(by_day.keys()):
        s = by_day[day]
        if not s:
            lines.append(f"  day {day}: insufficient rows")
            continue
        lines.append(
            f"  day {day}: lag1(price_dev)={s['lag1_price_dev']:.4f}, lag1(iv_res)={s['lag1_iv_res']:.4f}, "
            f"mean|price_dev|={s['mean_abs_price_dev']:.3f}, mean|iv_res|={s['mean_abs_iv_res']:.4f}, "
            f"mean(iv_res)={s['mean_iv_res']:.4f} (pos=rich vs smile), median_spread={s['median_spread']:.3f}, "
            f"median log(K/S)={s['median_log_KS']:.4f}"
        )
    lines.extend(["", "COMBINED (all days, pooled stats)", "-" * 78])
    if combined_rows and combined_rows[0]:
        cr = combined_rows[0]
        lines.append(
            f"  lag1(price_dev)={cr['lag1_price_dev']:.4f}, lag1(iv_res)={cr['lag1_iv_res']:.4f}, "
            f"mean|price_dev|={cr['mean_abs_price_dev']:.3f}, mean|iv_res|={cr['mean_abs_iv_res']:.4f}"
        )
    else:
        lines.append("  (insufficient pooled data)")

    # Strategy narrative from heuristics
    lag1_pd = [by_day[d]["lag1_price_dev"] for d in by_day if by_day[d] and np.isfinite(by_day[d]["lag1_price_dev"])]
    mean_bias = [by_day[d]["mean_iv_res"] for d in by_day if by_day[d]]
    mabs_pd = np.nanmean([by_day[d]["mean_abs_price_dev"] for d in by_day if by_day[d]]) if by_day else float("nan")
    med_sp = np.nanmean([by_day[d]["median_spread"] for d in by_day if by_day[d]]) if by_day else float("nan")

    lines.extend(["", "FRANKFURT-STYLE STRATEGY FIT", "-" * 78])
    avg_lag1_pd = float(np.nanmean(lag1_pd)) if lag1_pd else float("nan")
    avg_bias = float(np.nanmean(mean_bias)) if mean_bias else float("nan")

    iv_scalp_bits = []
    if avg_lag1_pd < 0.05:
        iv_scalp_bits.append(
            "Lag-1 autocorr of price deviation is not strongly positive on average — "
            "less evidence of one-directional stickiness; short-horizon IV / relative-value "
            "scalping is more plausible than if ACF were large positive."
        )
    elif avg_lag1_pd > 0.15:
        iv_scalp_bits.append(
            "Lag-1 autocorr of price deviation is meaningfully positive — deviations can persist; "
            "naive high-frequency mean-reversion vs smile-theo is riskier; size down or favor "
            "trend-aware exits."
        )
    else:
        iv_scalp_bits.append(
            "Moderate lag-1 structure in price deviations — treat IV scalping as conditional "
            "on spread and inventory, not as guaranteed oscillation."
        )

    if avg_bias > 0.01:
        iv_scalp_bits.append(
            "Mean IV residual vs neighbors is positive → often RICH vs the fitted smile; "
            "fade-rich / sell-vol style ideas deserve respect only after spread + limits check."
        )
    elif avg_bias < -0.01:
        iv_scalp_bits.append(
            "Mean IV residual negative → often CHEAP vs the smile; buy-vol / lift cheapness "
            "stories are the natural first read (still not automatic edge)."
        )
    else:
        iv_scalp_bits.append("Mean IV residual near zero vs smile — less systematic wing skew signal.")

    if med_sp > 0 and mabs_pd > 0 and med_sp > 0.5 * mabs_pd:
        iv_scalp_bits.append(
            "Median quoted spread is a large fraction of typical |price_dev| — microstructure "
            "may dominate; IV scalping needs passive or patient execution."
        )

    lines.append("IV scalping / relative-value (Frankfurt path): " + " ".join(iv_scalp_bits))

    lines.extend(
        [
            "",
            "Gamma / hybrid scaling: Prosperity writeups often pair options with underlying "
            "structure. This strike's log(K/S) path (see log_K_over_S_vs_time.png) shows how "
            "often you are near-the-money (gamma-sensitive) vs deep wing (directional/lottery). "
            "Use ../_common/underlying_fig08_day*.png for extract return persistence; hybrid "
            "ideas make more sense when ATM-ish AND underlying diagnostics show structure you trust.",
            "",
            "OTHER / BETTER-FIT STRATEGIES",
            "-" * 78,
            "- If spreads are tight vs |price_dev| and ACF is benign: prioritize relative-value "
            "scalping vs smile with tight risk.",
            "- If persistence dominates: consider slower reversion plays, inventory-biased quoting, "
            "or avoiding this name for HF straddle/scalp.",
            "- Deep wings with wide spreads: often better as occasional size when mispricing in "
            "PRICE is huge, not as baseline HF gamma farm.",
            "",
            "CROSS-REFERENCE",
            "-" * 78,
            "Cross-strike smile + near-money cloud: ../combined_analysis/ (iv_*.png, frankfurt_fig06a*.png).",
            "Fig 8 underlying ACF: ../_common/underlying_fig08_day*.png",
            "",
    ]
    )

    (voucher_path / "FINDINGS.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    apply_frankfurt_style()
    OUT_PV.mkdir(parents=True, exist_ok=True)
    print("Building resdf per day (full session) …")
    by_day_res: dict[int, pd.DataFrame] = {}
    for day in (0, 1, 2):
        by_day_res[day] = build_frankfurt_resdf_full_day(day)
        print(f"  day {day}: {len(by_day_res[day])} rows")

    plot_fig8_days_to_common()
    (OUT_PV / "_common" / "README.txt").write_text(
        "Underlying-only Figure 8 analogs (same for all VEV_*).\n"
        "See combined_analysis/ for the original bundled plots and full methodology text.\n",
        encoding="utf-8",
    )

    for voucher in VOUCHERS:
        K = int(voucher.split("_")[1])
        vdir = OUT_PV / voucher
        vdir.mkdir(parents=True, exist_ok=True)
        by_stats: dict[int, dict[str, float]] = {}
        for day in (0, 1, 2):
            ddir = vdir / f"day{day}"
            by_stats[day] = plot_day_bundle(by_day_res[day], day, voucher, ddir)

        cdir = vdir / "combined"
        cdir.mkdir(parents=True, exist_ok=True)
        parts = [by_day_res[d] for d in (0, 1, 2)]
        plot_combined_timeline(
            parts,
            voucher,
            cdir,
            "iv_res",
            "fig06b_iv_residual_concat.png",
            r"IV residual $v-\hat v$",
            f"Fig 6b analog — {voucher} — days 0–2 concatenated",
        )
        plot_combined_timeline(
            parts,
            voucher,
            cdir,
            "price_dev",
            "fig06c_price_dev_concat.png",
            "mid − BS(smile IV)",
            f"Fig 6c analog — {voucher} — days 0–2 concatenated",
        )
        allg = pd.concat(
            [by_day_res[d][by_day_res[d]["voucher"] == voucher] for d in (0, 1, 2)],
            ignore_index=True,
        ).sort_values(["day", "timestamp"])
        allg.to_csv(cdir / "resdf_slice_all_days.csv", index=False)
        combined_stats = {}
        if len(allg) > 20:
            combined_stats = {
                "lag1_price_dev": lag1_autocorr(allg["price_dev"]),
                "lag1_iv_res": lag1_autocorr(allg["iv_res"]),
                "mean_abs_price_dev": float(np.nanmean(np.abs(allg["price_dev"]))),
                "mean_abs_iv_res": float(np.nanmean(np.abs(allg["iv_res"]))),
            }
            pd.DataFrame([combined_stats]).to_csv(cdir / "stats_summary.csv", index=False)
        write_findings(vdir, voucher, K, by_stats, [combined_stats] if combined_stats else [])

    print("Wrote per-voucher tree under", OUT_PV)


if __name__ == "__main__":
    main()
