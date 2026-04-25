#!/usr/bin/env python3
"""
Round 3 tape analysis: BS implied vol, vega, delta from mids; spread; vega/|pos| rail stats.
TTE/DTE: round3work/round3description.txt + intraday winding per
round3work/plotting/original_method/combined_analysis/plot_iv_smile_round3.py
(csv day 0->8d open, 1->7d, 2->6d; DTE_eff = dte_open - (ts//100)/10000).
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_3"
OUT_DIR = Path(__file__).resolve().parent / "analysis_outputs"
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHERS = [f"VEV_{k}" for k in STRIKES]
COEFFS = [0.14215151147708086, -0.0016298611395181932, 0.23576325646627055]


def dte_from_csv_day(day: int) -> int:
    return 8 - int(day)


def dte_effective(day: int, timestamp: int) -> float:
    prog = (int(timestamp) // 100) / 10_000.0
    return max(float(dte_from_csv_day(day)) - prog, 1e-6)


def t_years(day: int, ts: int) -> float:
    return dte_effective(day, ts) / 365.0


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
    if market <= intrinsic + 1e-9 or market >= S - 1e-9 or S <= 0 or K <= 0 or T <= 0:
        return float("nan")

    def f(sig: float) -> float:
        return bs_call_price(S, K, T, sig, r) - market

    lo, hi = 1e-5, 15.0
    try:
        if f(lo) > 0 or f(hi) < 0:
            return float("nan")
        return brentq(f, lo, hi, xtol=1e-7, rtol=1e-7)
    except ValueError:
        return float("nan")


def iv_smile_model(S: float, K: float, T: float) -> float:
    if S <= 0 or K <= 0 or T <= 0:
        return float("nan")
    m = math.log(K / S) / math.sqrt(T)
    return float(np.polyval(np.asarray(COEFFS, dtype=float), m))


def bs_vega(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 0 or sigma <= 1e-12:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return float(S * float(norm.pdf(d1)) * math.sqrt(T))


def bs_delta_call(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 0 or sigma <= 1e-12:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return float(norm.cdf(d1))


def load_day_rows(day: int) -> dict[int, dict[str, dict]]:
    path = DATA / f"prices_round_3_day_{day}.csv"
    by_ts: dict[int, dict[str, dict]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            ts = int(row["timestamp"])
            prod = row["product"]
            by_ts[ts][prod] = row
    return by_ts


def row_mid_spread(row: dict) -> tuple[float | None, float | None]:
    bp = row.get("bid_price_1") or ""
    ap = row.get("ask_price_1") or ""
    if not bp or not ap:
        return None, None
    b, a = int(bp), int(ap)
    if a <= b:
        return None, None
    return 0.5 * (b + a), float(a - b)


def sample_analysis(max_ts_per_day: int = 120) -> dict:
    """Sparse timestamps per day for IV/vega/delta and spread stats."""
    out: dict = {
        "tte_assumption": (
            "Historical CSV day column 0,1,2 maps to DTE at session open 8,7,6 respectively "
            "(round3work/round3description.txt example: day index aligns with tutorial/R1/R2). "
            "Intraday: DTE_eff = DTE_open - (timestamp//100)/10000 (one day over ~10k hundred-steps), "
            "T = DTE_eff/365 years — same as plot_iv_smile_round3.py."
        ),
        "coeffs_high_to_low": COEFFS,
        "per_day": [],
    }
    for day in (0, 1, 2):
        by_ts = load_day_rows(day)
        timestamps = sorted(by_ts.keys())[:: max(1, len(by_ts) // max_ts_per_day)]
        vega_over_pos_sim: list[float] = []
        spreads: list[float] = []
        iv_mkt_minus_model: list[float] = []
        for ts in timestamps:
            rows = by_ts[ts]
            er = rows.get("VELVETFRUIT_EXTRACT")
            if not er:
                continue
            S_m, _ = row_mid_spread(er)
            if S_m is None:
                continue
            S = float(S_m)
            T = t_years(day, ts)
            for v in VOUCHERS:
                vr = rows.get(v)
                if not vr:
                    continue
                mid, sp = row_mid_spread(vr)
                if mid is None:
                    continue
                spreads.append(sp)
                iv_m = implied_iv = implied_vol_call(float(mid), S, int(v.split("_")[1]), T)
                iv_mod = iv_smile_model(S, int(v.split("_")[1]), T)
                if np.isfinite(iv_m) and np.isfinite(iv_mod):
                    iv_mkt_minus_model.append(float(iv_m - iv_mod))
                sig = iv_mod if np.isfinite(iv_mod) else (iv_m if np.isfinite(iv_m) else float("nan"))
                if not np.isfinite(sig):
                    continue
                K = int(v.split("_")[1])
                vega = bs_vega(S, K, T, sig)
                pos_hyp = 50
                if vega > 1e-9:
                    vega_over_pos_sim.append(vega / pos_hyp)
        out["per_day"].append(
            {
                "csv_day": day,
                "dte_open": dte_from_csv_day(day),
                "sampled_timestamps": len(timestamps),
                "mean_spread_l1": float(np.nanmean(spreads)) if spreads else None,
                "mean_iv_mkt_minus_model": float(np.nanmean(iv_mkt_minus_model)) if iv_mkt_minus_model else None,
                "median_vega_over_pos50": float(np.median(vega_over_pos_sim)) if vega_over_pos_sim else None,
            }
        )
    rails = [400.0, 600.0, 800.0, 1000.0, 1200.0]
    grid = []
    for rail in rails:
        caps = []
        for day in (0, 1, 2):
            by_ts = load_day_rows(day)
            ts0 = sorted(by_ts.keys())[len(by_ts) // 2]
            rows = by_ts[ts0]
            er = rows.get("VELVETFRUIT_EXTRACT")
            if not er:
                continue
            S_m, _ = row_mid_spread(er)
            if S_m is None:
                continue
            S = float(S_m)
            T = t_years(day, ts0)
            for v in VOUCHERS:
                vr = rows.get(v)
                if not vr:
                    continue
                mid, _ = row_mid_spread(vr)
                if mid is None:
                    continue
                K = int(v.split("_")[1])
                sig = iv_smile_model(S, K, T)
                if not np.isfinite(sig):
                    continue
                vega = bs_vega(S, K, T, sig)
                cap = min(300, int(rail / max(vega, 0.01)))
                caps.append({"v": v, "vega": vega, "cap_abs_pos": cap})
        grid.append({"vega_rail_constant": rail, "strike_caps_mid_session": caps})
    out["vega_rail_grid"] = grid
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = sample_analysis()
    out_path = OUT_DIR / "vega_iv_spread_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
