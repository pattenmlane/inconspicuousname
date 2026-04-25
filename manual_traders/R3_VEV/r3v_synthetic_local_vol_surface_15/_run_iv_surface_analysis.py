#!/usr/bin/env python3
"""One-off tape analysis: IV smile, vega-weighted structure, spreads — Round 3 days 0–2."""
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
OUT = Path(__file__).resolve().parent

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHERS = [f"VEV_{k}" for k in STRIKES]


def dte_from_csv_day(day: int) -> int:
    return 8 - int(day)


def intraday_progress(ts: int) -> float:
    return (int(ts) // 100) / 10_000.0


def dte_effective(day: int, ts: int) -> float:
    return max(float(dte_from_csv_day(day)) - intraday_progress(ts), 1e-6)


def t_years(day: int, ts: int) -> float:
    return dte_effective(day, ts) / 365.0


def bs_call(S: float, K: float, T: float, sig: float, r: float = 0.0) -> float:
    if T <= 0:
        return max(S - K, 0.0)
    if sig <= 1e-12:
        return max(S - K, 0.0)
    v = sig * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / v
    d2 = d1 - v
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def implied_vol(mid: float, S: float, K: float, T: float, r: float = 0.0) -> float:
    intrinsic = max(S - K, 0.0)
    if mid <= intrinsic + 1e-9 or mid >= S - 1e-9 or S <= 0 or K <= 0 or T <= 0:
        return float("nan")

    def f(sig: float) -> float:
        return bs_call(S, K, T, sig, r) - mid

    lo, hi = 1e-5, 15.0
    try:
        if f(lo) > 0 or f(hi) < 0:
            return float("nan")
        return brentq(f, lo, hi, xtol=1e-8, rtol=1e-8)
    except ValueError:
        return float("nan")


def vega(S: float, K: float, T: float, sig: float, r: float = 0.0) -> float:
    if T <= 0 or sig <= 1e-12:
        return 0.0
    v = sig * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / v
    return S * norm.pdf(d1) * math.sqrt(T)


def load_snapshots(day: int, step: int = 5000) -> list[dict]:
    path = DATA / f"prices_round_3_day_{day}.csv"
    rows_by_ts: dict[int, dict[str, dict]] = defaultdict(dict)
    with path.open(encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            ts = int(row["timestamp"])
            prod = row["product"]
            if prod == "VELVETFRUIT_EXTRACT" or prod in VOUCHERS:
                rows_by_ts[ts][prod] = row
    keys = sorted(rows_by_ts.keys())
    out = []
    for ts in keys[::step]:
        bundle = rows_by_ts[ts]
        if "VELVETFRUIT_EXTRACT" not in bundle:
            continue
        out.append({"ts": ts, "rows": bundle})
    return out


def mid_from_row(row: dict) -> tuple[float | None, float | None]:
    try:
        m = row.get("mid_price")
        if m is not None and m != "":
            return float(m), None
    except (TypeError, ValueError):
        pass
    bids = []
    asks = []
    for i in (1, 2, 3):
        bp, bv = row.get(f"bid_price_{i}"), row.get(f"bid_volume_{i}")
        ap, av = row.get(f"ask_price_{i}"), row.get(f"ask_volume_{i}")
        if bp not in (None, "") and bv not in (None, "") and int(float(bv)) > 0:
            bids.append(float(bp))
        if ap not in (None, "") and av not in (None, "") and int(float(av)) > 0:
            asks.append(float(ap))
    if not bids or not asks:
        return None, None
    bb, ba = max(bids), min(asks)
    return 0.5 * (bb + ba), 0.5 * (ba - bb)


def main() -> None:
    summaries = []
    all_iv_ranges = []
    for day in (0, 1, 2):
        snaps = load_snapshots(day, step=4000)
        iv_spreads = []
        vegas_atm = []
        for s in snaps[:200]:
            ts = s["ts"]
            T = t_years(day, ts)
            erow = s["rows"].get("VELVETFRUIT_EXTRACT")
            if not erow:
                continue
            S, _ = mid_from_row(erow)
            if S is None or S <= 0:
                continue
            ivs = []
            hs = []
            vegs = []
            for v in VOUCHERS:
                row = s["rows"].get(v)
                if not row:
                    continue
                mid, spr = mid_from_row(row)
                if mid is None:
                    continue
                k = int(v.split("_")[1])
                iv = implied_vol(mid, S, k, T, 0.0)
                if not math.isfinite(iv):
                    continue
                ivs.append(iv)
                hs.append(spr if spr is not None else 1.0)
                vegs.append(vega(S, k, T, iv, 0.0))
            if len(ivs) < 5:
                continue
            iv_spreads.append(max(ivs) - min(ivs))
            # ATM ~ closest strike to S
            j = min(range(len(STRIKES)), key=lambda i: abs(STRIKES[i] - S))
            vlab = VOUCHERS[j]
            row = s["rows"].get(vlab)
            if row:
                mid, _ = mid_from_row(row)
                if mid is not None:
                    k = STRIKES[j]
                    iv = implied_vol(mid, S, k, T, 0.0)
                    if math.isfinite(iv):
                        vegas_atm.append(vega(S, k, T, iv, 0.0))
        summaries.append(
            {
                "csv_day": day,
                "dte_open": dte_from_csv_day(day),
                "sampled_snapshots": len(snaps),
                "iv_smile_width_mean": float(np.nanmean(iv_spreads)) if iv_spreads else None,
                "iv_smile_width_p50": float(np.nanpercentile(iv_spreads, 50)) if iv_spreads else None,
                "half_spread_vouchers_mean": None,
                "vega_atm_mean": float(np.mean(vegas_atm)) if vegas_atm else None,
            }
        )
        all_iv_ranges.extend(iv_spreads)

    payload = {
        "timing": {
            "source": "round3work/round3description.txt + round3work/plotting/.../plot_iv_smile_round3.py",
            "csv_day_to_dte_at_open": {"0": 8, "1": 7, "2": 6},
            "intraday": "dte_eff = dte_open - (timestamp//100)/10000; T = dte_eff/365; r=0",
        },
        "bs_convention": "European call, implied vol from mid via Brent on [1e-5,15]",
        "per_day": summaries,
        "aggregate_iv_smile_width_p50": float(np.percentile(all_iv_ranges, 50)) if all_iv_ranges else None,
    }
    out_path = OUT / "iv_surface_tape_summary.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
