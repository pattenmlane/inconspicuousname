"""
Round 3 historical prices: implied vol vs moneyness (IV smile) for VEV_* on VELVETFRUIT_EXTRACT.

DTE: CSV `day` 0 -> 8d at open, 1 -> 7d, 2 -> 6d (see round3work/round3description.txt).
Intraday: DTE winds down ~1 day over the session (Frankfurt-style); T = dte_eff/365. r = 0.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

# Lives in plotting/original_method/combined_analysis/ → five parents to repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
DATA_DIR = REPO_ROOT / "Prosperity4Data" / "ROUND_3"
OUT_DIR = Path(__file__).resolve().parent


def dte_from_csv_day(day: int) -> int:
    """Calendar DTE at the start of that historical CSV day (before intraday winding)."""
    return 8 - int(day)


def intraday_progress(timestamp: int) -> float:
    """0 at session open → ~1 at session end (timestamps 0..999900 step 100 → 10_000 steps)."""
    return (int(timestamp) // 100) / 10_000.0


def dte_effective(day: int, timestamp: int) -> float:
    """Winding DTE: start-of-day calendar DTE minus ~one full day across the session."""
    return max(float(dte_from_csv_day(day)) - intraday_progress(timestamp), 1e-6)


def t_years_effective(day: int, timestamp: int) -> float:
    return dte_effective(day, timestamp) / 365.0


def bs_call_price(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 1e-12:
        return max(S - K, 0.0)
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / v
    d2 = d1 - v
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def implied_vol_call(market: float, S: float, K: float, T: float, r: float = 0.0) -> float:
    intrinsic = max(S - K, 0.0)
    if market <= intrinsic + 1e-9:
        return float("nan")
    if market >= S - 1e-9:
        return float("nan")
    if S <= 0 or K <= 0 or T <= 0:
        return float("nan")

    def f(sig: float) -> float:
        return bs_call_price(S, K, T, sig, r) - market

    lo, hi = 1e-5, 15.0
    try:
        fl, fh = f(lo), f(hi)
        if fl > 0:
            return float("nan")
        if fh < 0:
            return float("nan")
        return brentq(f, lo, hi, xtol=1e-8, rtol=1e-8)
    except ValueError:
        return float("nan")


STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHERS = [f"VEV_{k}" for k in STRIKES]


def load_day_wide(day: int) -> pd.DataFrame:
    path = DATA_DIR / f"prices_round_3_day_{day}.csv"
    df = pd.read_csv(path, sep=";")
    pvt = df.pivot_table(
        index="timestamp",
        columns="product",
        values="mid_price",
        aggfunc="first",
    )
    if "VELVETFRUIT_EXTRACT" not in pvt.columns:
        raise RuntimeError("missing underlying column")
    vcols = [v for v in VOUCHERS if v in pvt.columns]
    out = pvt[["VELVETFRUIT_EXTRACT"] + vcols].copy()
    out.columns = ["S"] + vcols
    out["day"] = day
    out["dte"] = dte_from_csv_day(day)
    return out


def subsample_wide(wide: pd.DataFrame, step: int = 20) -> pd.DataFrame:
    n = len(wide)
    pos = np.unique(np.r_[0, n // 2, n - 1, np.arange(0, n, step, dtype=int)])
    return wide.iloc[pos].copy()


def compute_iv_panel(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ts, row in wide.iterrows():
        S = float(row["S"])
        day_i = int(row["day"])
        ts_i = int(ts)
        ty = t_years_effective(day_i, ts_i)
        d_eff = dte_effective(day_i, ts_i)
        for v in VOUCHERS:
            if v not in row.index:
                continue
            K = int(v.split("_")[1])
            mid = float(row[v])
            iv = implied_vol_call(mid, S, K, ty, 0.0)
            mny = K / S if S > 0 else float("nan")
            log_mny = math.log(K / S) if S > 0 and K > 0 else float("nan")
            log_sk = math.log(S / K) if S > 0 and K > 0 else float("nan")
            intrinsic = max(S - K, 0.0)
            rows.append(
                {
                    "timestamp": ts_i,
                    "day": day_i,
                    "dte": int(row["dte"]),
                    "dte_eff": float(d_eff),
                    "t_years": float(ty),
                    "voucher": v,
                    "K": K,
                    "S": S,
                    "mid": mid,
                    "intrinsic": intrinsic,
                    "time_value": mid - intrinsic,
                    "moneyness_K_over_S": mny,
                    "log_moneyness": log_mny,
                    "log_S_over_K": log_sk,
                    "iv": iv,
                }
            )
    return pd.DataFrame(rows)


def fit_smile_poly(
    g: pd.DataFrame, xcol: str, ycol: str = "iv"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quadratic IV smile in an arbitrary moneyness column (e.g. log(K/S) or log(S/K))."""
    x = g[xcol].to_numpy(dtype=float)
    y = g[ycol].to_numpy(dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x_fit, y_fit = x[m], y[m]
    if len(x_fit) < 4:
        return np.full(len(g), np.nan), np.full(len(g), np.nan), np.array([])
    coeffs = np.polyfit(x_fit, y_fit, 2)
    xf = g[xcol].to_numpy(dtype=float)
    fit = np.polyval(coeffs, xf)
    res = g[ycol].to_numpy(dtype=float) - fit
    return fit, res, coeffs


def fit_smile_residuals(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return fit_smile_poly(g, "log_moneyness", "iv")


def build_iv_and_residual_dataframe(step: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (ivdf, resdf) for downstream analysis."""
    all_iv = []
    for day in (0, 1, 2):
        wide = load_day_wide(day)
        panel = compute_iv_panel(subsample_wide(wide, step=step))
        all_iv.append(panel)
    ivdf = pd.concat(all_iv, ignore_index=True)
    res_records = []
    for day in (0, 1, 2):
        sub = ivdf[ivdf["day"] == day]
        for ts in sub["timestamp"].unique():
            g = sub[sub["timestamp"] == ts].copy()
            fit, res, _ = fit_smile_residuals(g)
            g = g.assign(iv_fit=fit, iv_res=res)
            res_records.append(g)
    resdf = pd.concat(res_records, ignore_index=True)
    return ivdf, resdf


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ivdf, resdf = build_iv_and_residual_dataframe(step=20)

    snap_times = [0, 500_000, 999_900]
    strike_to_color = {k: plt.cm.tab10(i / 9.0) for i, k in enumerate(STRIKES)}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for ax, day in zip(axes, (0, 1, 2)):
        sub = ivdf[ivdf["day"] == day]
        for t in snap_times:
            g = sub[sub["timestamp"] == t].sort_values("K")
            ax.plot(g["log_moneyness"], g["iv"], "o-", alpha=0.85, markersize=5, label=f"t={t}")
        ax.set_title(f"day={day} (DTE open={dte_from_csv_day(day)}, winds intraday)")
        ax.set_xlabel("log(K/S)")
        ax.set_ylabel("implied vol")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("IV smile snapshots (early / mid / late session)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "iv_smile_snapshots.png", dpi=160)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for ax, day in zip(axes, (0, 1, 2)):
        sub = ivdf[ivdf["day"] == day]
        for k in STRIKES:
            g = sub[sub["K"] == k]
            ax.scatter(
                g["log_moneyness"],
                g["iv"],
                s=4,
                alpha=0.35,
                c=[strike_to_color[k]] * len(g),
                label=f"{k}" if day == 0 else None,
            )
        ax.set_title(f"day={day} DTE open={dte_from_csv_day(day)} (subsampled, winds)")
        ax.set_xlabel("log(K/S)")
        ax.set_ylabel("IV")
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=7, ncol=2)
    fig.suptitle("IV vs log(K/S)", fontsize=12)
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
    ax.set_title("Median IV by strike")
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
        ax.scatter(rr["log_moneyness"], rr["iv_res"], s=6, alpha=0.25, label=f"day {day}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("log(K/S)")
    ax.set_ylabel("IV − parabolic fit")
    ax.set_title("Detrended IV (Frankfurt-style)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "iv_residuals_detrended.png", dpi=160)
    plt.close()

    ivdf.sample(min(50000, len(ivdf)), random_state=0).to_csv(
        OUT_DIR / "iv_panel_sample.csv", index=False
    )

    print("Wrote plots and tables to", OUT_DIR)


if __name__ == "__main__":
    main()
