#!/usr/bin/env python3
"""
Offline Round-3 tape analysis: implied vol (Black–Scholes call) and vega
for VEV strikes, using round3work/round3description.txt TTE mapping.

Historical CSV day column 0,1,2 maps to TTE 8d, 7d, 6d (same offset as doc
example: tutorial round = +1 day vs final sim TTE).
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import NormalDist

_N = NormalDist()
ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "Prosperity4Data" / "ROUND_3"
OUT_DIR = Path(__file__).resolve().parent / "analysis_outputs"
VEV = [
    "VEV_4000",
    "VEV_4500",
    "VEV_5000",
    "VEV_5100",
    "VEV_5200",
    "VEV_5300",
    "VEV_5400",
    "VEV_5500",
    "VEV_6000",
    "VEV_6500",
]


def tte_days(hist_day: int) -> float:
    """From round3description: hist day i lines up with TTE = 8 - i for i in 0,1,2."""
    return float(8 - hist_day)


def bs_call_price(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(S - K, 0.0)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return S * _N.cdf(d1) - K * math.exp(-r * T) * _N.cdf(d2)


def bs_vega(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    return S * sqrtT * _N.pdf(d1)


def implied_vol_bisect(mid: float, S: float, K: float, T: float, lo=0.01, hi=3.0) -> float | None:
    if mid <= 0 or S <= 0 or T <= 0:
        return None
    intrinsic = max(S - K, 0.0)
    if mid < intrinsic - 1e-6:
        return None
    for _ in range(60):
        sig = 0.5 * (lo + hi)
        p = bs_call_price(S, K, T, sig)
        if p > mid:
            hi = sig
        else:
            lo = sig
    return 0.5 * (lo + hi)


def spread_width(row: dict) -> float | None:
    bp = row.get("bid_price_1")
    ap = row.get("ask_price_1")
    if bp is None or ap is None:
        return None
    return float(ap) - float(bp)


def load_last_mids(path: Path, products: set[str], max_ts: int | None = None) -> dict[str, tuple[float, float, int]]:
    """product -> (mid, spread, timestamp) at last seen row (optionally capped ts)."""
    best: dict[str, tuple[float, float, int]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            ts = int(row["timestamp"])
            if max_ts is not None and ts > max_ts:
                continue
            prod = row["product"]
            if prod not in products:
                continue
            mid = float(row["mid_price"])
            sp = spread_width(row) or 0.0
            best[prod] = (mid, sp, ts)
    return best


def main():
    products = set(VEV) | {"VELVETFRUIT_EXTRACT"}
    summary: dict = {"tte_note": "hist_day d in {0,1,2} -> TTE = 8-d days per round3description.txt example offset."}
    per_day = []

    for hist_day in (0, 1, 2):
        csv_path = DATA_DIR / f"prices_round_3_day_{hist_day}.csv"
        T_years = tte_days(hist_day) / 365.0
        mids = load_last_mids(csv_path, products, max_ts=500_000)
        S = mids.get("VELVETFRUIT_EXTRACT", (None,))[0]
        day_block: dict = {"hist_day": hist_day, "tte_days": tte_days(hist_day), "underlying_mid_sample_ts": None}
        if S is None:
            day_block["error"] = "no extract"
            per_day.append(day_block)
            continue
        day_block["underlying_S"] = S
        ivs = {}
        vegas = {}
        spreads = {}
        for sym in VEV:
            if sym not in mids:
                continue
            mid, spr, ts = mids[sym]
            K = float(sym.split("_")[1])
            iv = implied_vol_bisect(mid, S, K, T_years)
            if iv is None:
                continue
            ivs[sym] = round(iv, 4)
            vegas[sym] = round(bs_vega(S, K, T_years, iv), 4)
            spreads[sym] = round(spr, 2)
        day_block["implied_vol_atm_region"] = {k: ivs[k] for k in ("VEV_5000", "VEV_5100") if k in ivs}
        day_block["iv_all"] = ivs
        day_block["vega"] = vegas
        day_block["spread_l1"] = spreads
        m5 = mids.get("VEV_5000", (0,))[0]
        m51 = mids.get("VEV_5100", (0,))[0]
        if m5 and m51:
            day_block["straddle_mid_sum"] = round(m5 + m51, 2)
            c5 = bs_call_price(S, 5000.0, T_years, ivs.get("VEV_5000", 0.3))
            c51 = bs_call_price(S, 5100.0, T_years, ivs.get("VEV_5100", 0.3))
            day_block["bs_theo_straddle_same_iv_as_each_leg"] = "use per-leg IV above separately"
            day_block["vega_straddle"] = round(
                vegas.get("VEV_5000", 0) + vegas.get("VEV_5100", 0), 4
            )
        per_day.append(day_block)

    summary["by_day"] = per_day
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "iv_vega_spread_snapshot.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
