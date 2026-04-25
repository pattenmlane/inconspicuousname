#!/usr/bin/env python3
"""
Round 3 tape analysis: IV smile, vega-weighted structure, spreads vs extract |ΔS|/z.
TTE/DTE: round3work/round3description.txt + intraday wind from plot_iv_smile_round3
(csv day 0 -> 8d open, 1->7, 2->6; DTE_eff = dte_open - progress with progress = (ts//100)/10000).
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_3"
OUT_DIR = Path(__file__).resolve().parent / "analysis_outputs"

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHERS = [f"VEV_{k}" for k in STRIKES]


def dte_from_csv_day(day: int) -> int:
    return 8 - int(day)


def intraday_progress(timestamp: int) -> float:
    return (int(timestamp) // 100) / 10_000.0


def dte_effective(day: int, timestamp: int) -> float:
    return max(float(dte_from_csv_day(day)) - intraday_progress(timestamp), 1e-6)


def t_years(day: int, timestamp: int) -> float:
    return dte_effective(day, timestamp) / 365.0


def bs_call(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 0 or sigma <= 1e-12:
        return max(S - K, 0.0)
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / v
    d2 = d1 - v
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def implied_vol(market: float, S: float, K: float, T: float) -> float:
    intrinsic = max(S - K, 0.0)
    if market <= intrinsic + 1e-6 or S <= 0 or K <= 0 or T <= 0:
        return float("nan")
    if market >= S - 1e-6:
        return float("nan")

    def f(sig: float) -> float:
        return bs_call(S, K, T, sig) - market

    lo, hi = 1e-5, 15.0
    try:
        if f(lo) > 0 or f(hi) < 0:
            return float("nan")
        return float(brentq(f, lo, hi, xtol=1e-7, rtol=1e-7))
    except ValueError:
        return float("nan")


def bs_vega(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 0 or sigma <= 1e-12:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return float(S * float(norm.pdf(d1)) * math.sqrt(T))


def load_day(day: int) -> pd.DataFrame:
    p = DATA / f"prices_round_3_day_{day}.csv"
    return pd.read_csv(p, sep=";")


def main() -> None:
    random.seed(42)
    np.random.seed(42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows_out: list[dict] = []
    for day in (0, 1, 2):
        df = load_day(day)
        wide = df.pivot_table(index="timestamp", columns="product", values="mid_price", aggfunc="first")
        if "VELVETFRUIT_EXTRACT" not in wide.columns:
            continue
        Sser = wide["VELVETFRUIT_EXTRACT"].astype(float)
        ts_list = sorted(Sser.dropna().index.tolist())
        # subsample for runtime
        if len(ts_list) > 2500:
            ts_list = sorted(random.sample(ts_list, 2500))

        dlog: list[float] = []
        prev_s = None
        z_scores: list[float] = []
        win = 400
        buf: list[float] = []

        for ts in ts_list:
            s = float(Sser.loc[ts])
            if prev_s is not None and prev_s > 0:
                buf.append(math.log(s / prev_s))
            prev_s = s
            if len(buf) > win:
                buf = buf[-win:]
            if len(buf) >= 30:
                m = float(np.mean(buf))
                v = float(np.std(buf)) + 1e-12
                z_scores.append((abs(buf[-1] - m)) / v)
            else:
                z_scores.append(0.0)

        Tmap = {ts: t_years(day, ts) for ts in ts_list}

        for i, ts in enumerate(ts_list):
            S = float(Sser.loc[ts])
            if not math.isfinite(S) or S <= 0:
                continue
            T = Tmap[ts]
            ivs: dict[str, float] = {}
            spreads: dict[str, float] = {}
            sub = df[(df["timestamp"] == ts) & (df["product"].isin(VOUCHERS))]
            for _, r in sub.iterrows():
                pr = str(r["product"])
                m = float(r["mid_price"])
                k = int(pr.split("_")[1])
                iv = implied_vol(m, S, k, T)
                ivs[pr] = iv
                b1 = r.get("bid_price_1")
                a1 = r.get("ask_price_1")
                if pd.notna(b1) and pd.notna(a1):
                    spreads[pr] = float(a1) - float(b1)
                else:
                    spreads[pr] = float("nan")

            vals = [(k, ivs[f"VEV_{k}"]) for k in STRIKES if math.isfinite(ivs.get(f"VEV_{k}", float("nan")))]
            if len(vals) < 6:
                continue
            ks_sorted = sorted([k for k, _ in vals])
            iv_by_k = {k: v for k, v in vals}
            atm_k = min(STRIKES, key=lambda k: abs(float(k) - S))
            if atm_k not in iv_by_k:
                continue
            iv_atm = iv_by_k[atm_k]
            otm_calls = [iv_by_k[k] for k in ks_sorted if k > S and k in iv_by_k]
            otm_puts_side = [iv_by_k[k] for k in ks_sorted if k < S and k in iv_by_k]
            skew = float(np.nanmean(otm_calls) - np.nanmean(otm_puts_side)) if otm_calls and otm_puts_side else float("nan")
            smile_std = float(np.nanstd([iv_by_k[k] for k in ks_sorted if k in iv_by_k]))

            vegas = []
            for kk in STRIKES:
                pr = f"VEV_{kk}"
                sig = iv_by_k.get(pr, float("nan"))
                if math.isfinite(sig) and sig > 0:
                    vegas.append(bs_vega(S, float(kk), T, sig))
                else:
                    vegas.append(0.0)
            vtot = sum(vegas) + 1e-12
            w_atm = vegas[STRIKES.index(atm_k)] / vtot

            spr_atm = spreads.get(f"VEV_{atm_k}", float("nan"))
            z = z_scores[i] if i < len(z_scores) else 0.0

            rows_out.append(
                {
                    "csv_day": day,
                    "dte_open": dte_from_csv_day(day),
                    "timestamp": int(ts),
                    "S": S,
                    "z_abs_ret": float(z),
                    "iv_atm": float(iv_atm),
                    "skew_call_minus_put_moneyness": skew,
                    "iv_cross_std": smile_std,
                    "vega_weight_atm": float(w_atm),
                    "spread_atm": spr_atm,
                }
            )

    out_df = pd.DataFrame(rows_out)
    path_panel = OUT_DIR / "smile_panel_sampled.csv"
    path_csv = OUT_DIR / "smile_summary_by_day.csv"
    out_df.to_csv(path_panel, index=False)
    corrs = []
    for d in (0, 1, 2):
        sub = out_df[out_df["csv_day"] == d]
        if len(sub) > 20:
            c = sub["z_abs_ret"].corr(sub["iv_cross_std"])
            corrs.append({"csv_day": d, "corr_z_iv_cross_std": float(c)})
    corr_df = pd.DataFrame(corrs)
    g2 = out_df.groupby("csv_day")[["spread_atm", "iv_atm", "z_abs_ret"]].mean()
    g2.to_csv(path_csv)

    summary = {
        "timing_assumptions": (
            "round3description.txt: vouchers 7d from round1; historical example maps csv day to TTE at open "
            "(day0->8d, day1->7d, day2->6d). Intraday: DTE_eff = dte_open - (timestamp//100)/10000 "
            "(same as round3work/plotting/original_method/combined_analysis/plot_iv_smile_round3.py)."
        ),
        "mean_by_csv_day": g2.round(6).to_dict(),
        "corr_z_vs_iv_cross_std": corr_df.to_dict(orient="records"),
        "panel_rows": len(out_df),
        "artifacts": [str(path_panel.relative_to(REPO)), str(path_csv.relative_to(REPO))],
    }
    Path(OUT_DIR / "smile_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
