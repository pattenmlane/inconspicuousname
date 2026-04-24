"""
IV smile pipeline using **true_fv** (hold-1 log recovery) instead of tape mids.

TTE for these day-39 probes: **5 days at session open** (Round 3 final sim; round3description.txt).
Winding branch: same intraday fraction as historical Frankfurt style, applied to DTE=5.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[4]
_ORIG_COMBINED = _REPO / "round3work" / "plotting" / "original_method" / "combined_analysis"
sys.path.insert(0, str(_ORIG_COMBINED))

from plot_iv_smile_round3 import (  # noqa: E402
    STRIKES,
    VOUCHERS,
    fit_smile_poly,
    implied_vol_call,
)

from true_fv_loader import fv_mid_gap_summary, load_true_fv_wide  # noqa: E402

DTE_OPEN_ROUND3_PROBE = 5


def intraday_progress(timestamp: int) -> float:
    return (int(timestamp) // 100) / 10_000.0


def make_t_years_fn(winding: bool) -> Callable[[int], float]:
    def t_years(ts: int) -> float:
        if winding:
            d_eff = max(float(DTE_OPEN_ROUND3_PROBE) - intraday_progress(ts), 1e-6)
        else:
            d_eff = float(DTE_OPEN_ROUND3_PROBE)
        return d_eff / 365.0

    return t_years


def compute_iv_panel_fv(wide: pd.DataFrame, t_years_fn: Callable[[int], float]) -> pd.DataFrame:
    rows = []
    for ts, row in wide.iterrows():
        S = float(row["S"])
        ts_i = int(ts)
        ty = float(t_years_fn(ts_i))
        d_eff = ty * 365.0
        day_i = int(row["day"])
        for v in VOUCHERS:
            if v not in row.index:
                continue
            K = int(v.split("_")[1])
            fv = float(row[v])
            iv = implied_vol_call(fv, S, K, ty, 0.0)
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
                    "t_years": ty,
                    "voucher": v,
                    "K": K,
                    "S": S,
                    "fv": fv,
                    "mid": fv,
                    "intrinsic": intrinsic,
                    "time_value": fv - intrinsic,
                    "moneyness_K_over_S": mny,
                    "log_moneyness": log_mny,
                    "log_S_over_K": log_sk,
                    "iv": iv,
                }
            )
    return pd.DataFrame(rows)


def build_iv_and_residual_fv(wide: pd.DataFrame, t_years_fn: Callable[[int], float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ivdf = compute_iv_panel_fv(wide, t_years_fn)
    res_records = []
    sub = ivdf
    for ts in sorted(sub["timestamp"].unique()):
        g = sub[sub["timestamp"] == ts].copy()
        fit, res, _ = fit_smile_poly(g, "log_moneyness", "iv")
        g = g.assign(iv_fit=fit, iv_res=res)
        res_records.append(g)
    resdf = pd.concat(res_records, ignore_index=True)
    return ivdf, resdf


def global_quadratic_fit(ivdf: pd.DataFrame, xcol: str = "log_moneyness") -> tuple[np.ndarray, float, int]:
    """Frankfurt-style m_t = log(K/S)/sqrt(T) pool (same as voucher_work fit)."""
    xs: list[float] = []
    ys: list[float] = []
    for _, r in ivdf.iterrows():
        T = float(r["t_years"])
        if T <= 0 or not np.isfinite(T):
            continue
        S, K = float(r["S"]), float(r["K"])
        fv = float(r["fv"])
        iv = implied_vol_call(fv, S, K, T, 0.0)
        if not np.isfinite(iv):
            continue
        m_t = math.log(K / S) / math.sqrt(T)
        if not np.isfinite(m_t):
            continue
        xs.append(m_t)
        ys.append(iv)
    xf = np.asarray(xs, dtype=float)
    yf = np.asarray(ys, dtype=float)
    m = np.isfinite(xf) & np.isfinite(yf)
    xf, yf = xf[m], yf[m]
    if len(xf) < 30:
        return np.array([]), float("nan"), 0
    coeff = np.polyfit(xf, yf, 2)
    resid = yf - np.polyval(coeff, xf)
    rmse = float(np.sqrt(np.mean(resid**2)))
    return coeff, rmse, int(len(xf))


def run_all(winding: bool, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    wide = load_true_fv_wide(_REPO)
    tfn = make_t_years_fn(winding)
    ivdf, resdf = build_iv_and_residual_fv(wide, tfn)
    coeff, rmse, n_g = global_quadratic_fit(ivdf)

    mode = "wind_down" if winding else "no_wind_down"
    meta = {
        "mode": mode,
        "dte_open_probe": DTE_OPEN_ROUND3_PROBE,
        "day_index": 39,
        "n_timestamps": int(wide.shape[0]),
        "global_m_t_poly_coeffs_high_to_low": [float(c) for c in coeff] if len(coeff) else [],
        "global_m_t_fit_rmse": rmse,
        "global_m_t_fit_n": n_g,
    }
    (out_dir / "true_fv_iv_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    fv_mid_gap_summary(wide).to_csv(out_dir / "fv_vs_mid_mean_abs_gap.csv", index=False)

    snap_times = [0, 50_000, 99_900]
    strike_to_color = {k: plt.cm.tab10(i / 9.0) for i, k in enumerate(STRIKES)}

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sub = ivdf
    for t in snap_times:
        g = sub[sub["timestamp"] == t].sort_values("K")
        if len(g) < 3:
            continue
        ax.plot(g["log_moneyness"], g["iv"], "o-", alpha=0.85, markersize=5, label=f"t={t}")
    ax.set_title(
        f"IV smile snapshots (true_fv) — {mode}\nDTE_open={DTE_OPEN_ROUND3_PROBE}d"
        + (" (intraday wind)" if winding else " (no wind)")
    )
    ax.set_xlabel("log(K/S)")
    ax.set_ylabel("implied vol")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "iv_smile_snapshots.png", dpi=160)
    plt.close()

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for k in STRIKES:
        g = sub[sub["K"] == k]
        ax.scatter(
            g["log_moneyness"],
            g["iv"],
            s=4,
            alpha=0.35,
            c=[strike_to_color[k]] * len(g),
            label=f"{k}",
        )
    ax.set_title(f"IV vs log(K/S) — true_fv — {mode}")
    ax.set_xlabel("log(K/S)")
    ax.set_ylabel("IV")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "iv_vs_logmoneyness_scatter.png", dpi=160)
    plt.close()

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    m = sub.groupby("K")["iv"].median()
    ax.plot(m.index, m.values, "o-", label="median IV (true_fv)")
    ax.set_xlabel("strike K")
    ax.set_ylabel("median IV")
    ax.set_title(f"Median IV by strike — {mode}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "median_iv_by_strike.png", dpi=160)
    plt.close()

    summary = resdf.groupby(["day", "voucher"])["iv_res"].agg(
        mean="mean",
        std="std",
        mean_abs_res=lambda s: float(np.nanmean(np.abs(s))),
    )
    summary.to_csv(out_dir / "iv_smile_residual_summary.csv")

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ts_list = sorted(resdf["timestamp"].unique())[::5]
    rr = resdf[resdf["timestamp"].isin(ts_list)]
    ax.scatter(rr["log_moneyness"], rr["iv_res"], s=8, alpha=0.3, c="#1f77b4", label="subsampled ts")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("log(K/S)")
    ax.set_ylabel("IV − parabolic fit")
    ax.set_title(f"Detrended IV — true_fv — {mode}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "iv_residuals_detrended.png", dpi=160)
    plt.close()

    ivdf.sample(min(50_000, len(ivdf)), random_state=0).to_csv(out_dir / "iv_panel_sample.csv", index=False)
    print("Wrote", out_dir)
