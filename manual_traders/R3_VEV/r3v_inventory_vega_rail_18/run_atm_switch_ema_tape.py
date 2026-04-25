#!/usr/bin/env python3
"""Replicate v25-style EMA of |theo_diff - mean_theo| for ATM VEV_5100 (representative) per day; guides IV switch gate sweeps."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_3"
OUT = REPO / "manual_traders/R3_VEV/r3v_inventory_vega_rail_18/analysis_outputs/atm_switch_ema_5100_by_day.json"

COEFF = [0.14215151147708086, -0.0016298611395181932, 0.23576325646627055]
K = 5100
V = f"VEV_{K}"
THEO_NORM_WINDOW = 20
IV_SCALPING_WINDOW = 100


def t_years(day: int, ts: int) -> float:
    return max(float(8 - day) - (ts // 100) / 10_000.0, 1e-6) / 365.0


def iv_smile(S: float, T: float) -> float:
    m = math.log(K / S) / math.sqrt(T)
    return float(np.polyval(np.array(COEFF, dtype=float), m))


def bs_call(S: float, T: float, sig: float) -> float:
    if T <= 0 or sig <= 1e-12:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + 0.5 * sig**2 * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    return S * float(norm.cdf(d1)) - K * float(norm.cdf(d2))


def ema_value(old: float, window: int, value: float) -> float:
    a = 2.0 / (window + 1.0)
    return a * value + (1.0 - a) * old


def main() -> None:
    by_day: dict = {}
    for day in (0, 1, 2):
        theo_diff: float = 0.0
        switch: float = 0.0
        vals: list[float] = []
        by_ts: dict[int, dict] = defaultdict(dict)
        with (DATA / f"prices_round_3_day_{day}.csv").open() as f:
            r = csv.DictReader(f, delimiter=";")
            for row in r:
                by_ts[int(row["timestamp"])][row["product"]] = row
        for ts in sorted(by_ts):
            d = by_ts[ts]
            u = d.get("VELVETFRUIT_EXTRACT")
            rr = d.get(V)
            if not u or not u.get("bid_price_1") or not u.get("ask_price_1"):
                continue
            if not rr or not rr.get("bid_price_1") or not rr.get("ask_price_1"):
                continue
            S = 0.5 * (int(u["bid_price_1"]) + int(u["ask_price_1"]))
            if K / S < 0.97 or K / S > 1.03:
                continue
            T = t_years(day, ts)
            sig = iv_smile(S, T)
            if not math.isfinite(sig) or sig <= 0:
                continue
            theo = bs_call(S, T, sig)
            wall = 0.5 * (int(rr["bid_price_1"]) + int(rr["ask_price_1"]))
            diff = wall - theo
            theo_diff = ema_value(theo_diff, THEO_NORM_WINDOW, diff)
            switch = ema_value(switch, IV_SCALPING_WINDOW, abs(diff - theo_diff))
            vals.append(switch)
        a = np.array(vals, dtype=float)
        by_day[str(day)] = {
            "n": len(vals),
            "p50": float(np.quantile(a, 0.5)) if len(vals) else None,
            "p75": float(np.quantile(a, 0.75)) if len(vals) else None,
            "p90": float(np.quantile(a, 0.9)) if len(vals) else None,
            "p95": float(np.quantile(a, 0.95)) if len(vals) else None,
        }
    out = {
        "product": V,
        "method": "Same COEFF/BS/synthetic mid as v25; ATM 0.97-1.03; EMA windows 20/100.",
        "by_csv_day": by_day,
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
