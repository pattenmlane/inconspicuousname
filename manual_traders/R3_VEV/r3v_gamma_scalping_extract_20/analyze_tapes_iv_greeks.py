#!/usr/bin/env python3
"""
Round-3 tape analysis: Black–Scholes implied vol and Greeks from mids.

TTE mapping (from round3work/round3description.txt): vouchers have a 7-day
deadline from round 1; each round is one day. The spec’s worked example maps
historical tape days to competition rounds / TTE. For Round-3 historical tapes
labeled day 0, 1, 2 we use the same offset pattern as that example:
  historical day d  ->  TTE = (8 - d) days  (d=0 -> 8d, d=1 -> 7d, d=2 -> 6d).

Year convention: T in years = TTE_days / 365 (calendar days per spec).

Outputs under analysis_outputs/ (CSV + JSON summary).
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

_ROOT = Path(__file__).resolve().parents[3]
_TAPES = _ROOT / "Prosperity4Data" / "ROUND_3"
_OUT = Path(__file__).resolve().parent / "analysis_outputs"

VEV_PREFIX = "VEV_"
UNDER = "VELVETFRUIT_EXTRACT"
# Align with round3description example: tape day index d -> TTE days
TTE_DAYS_BY_TAPE_DAY = {0: 8, 1: 7, 2: 6}
DAYS_PER_YEAR = 365.0
_N = NormalDist()


def _cdf(x: float) -> float:
    return _N.cdf(x)


def _pdf(x: float) -> float:
    return _N.pdf(x)


def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    sig_sqrt_t = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / sig_sqrt_t
    d2 = d1 - sig_sqrt_t
    return S * _cdf(d1) - K * math.exp(-r * T) * _cdf(d2)


def bs_call_delta_gamma(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    if T <= 0 or sigma <= 0:
        return (1.0 if S > K else 0.0, 0.0)
    sig_sqrt_t = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / sig_sqrt_t
    delta = _cdf(d1)
    gamma = _pdf(d1) / (S * sig_sqrt_t)
    return delta, gamma


def implied_vol_bisect(mid: float, S: float, K: float, T: float, r: float) -> float | None:
    """IV (annualized) from European call mid; None if no bracket."""
    if mid <= 0 or S <= 0 or K <= 0 or T <= 0:
        return None
    intrinsic = max(S - K, 0.0)
    if mid < intrinsic - 1e-6:
        return None
    lo, hi = 1e-6, 5.0
    for _ in range(60):
        mid_sig = 0.5 * (lo + hi)
        p = bs_call_price(S, K, T, r, mid_sig)
        if p > mid:
            hi = mid_sig
        else:
            lo = mid_sig
    return 0.5 * (lo + hi)


def load_prices_by_ts(path: Path) -> dict[int, dict[str, dict]]:
    """timestamp -> product -> {mid, bid, ask, spread}."""
    by_ts: dict[int, dict[str, dict]] = defaultdict(dict)
    with path.open(encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            ts = int(row["timestamp"])
            prod = row["product"]
            mid = float(row["mid_price"])
            bp = row.get("bid_price_1") or ""
            ap = row.get("ask_price_1") or ""
            bid = int(bp) if bp != "" else None
            ask = int(ap) if ap != "" else None
            spread = (ask - bid) if (bid is not None and ask is not None) else None
            by_ts[ts][prod] = {"mid": mid, "bid": bid, "ask": ask, "spread": spread}
    return by_ts


def main() -> None:
    r = 0.0
    step = 500  # subsample timestamps (full tape has ~10k steps/day)
    rows_out: list[dict] = []
    iv_by_day_strike: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    spread_by_day_strike: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    gamma_net_by_day: dict[int, list[float]] = defaultdict(list)

    for tape_day in (0, 1, 2):
        csv_path = _TAPES / f"prices_round_3_day_{tape_day}.csv"
        data = load_prices_by_ts(csv_path)
        tte_days = TTE_DAYS_BY_TAPE_DAY[tape_day]
        T = tte_days / DAYS_PER_YEAR
        timestamps = sorted(data.keys())
        prev_S: float | None = None
        for ts in timestamps[::step]:
            prods = data[ts]
            if UNDER not in prods:
                continue
            S = prods[UNDER]["mid"]
            net_gamma = 0.0
            for prod, q in prods.items():
                if not prod.startswith(VEV_PREFIX):
                    continue
                K = int(prod.split("_", 1)[1])
                mid = q["mid"]
                iv = implied_vol_bisect(mid, S, K, T, r)
                if iv is None:
                    continue
                dlt, gam = bs_call_delta_gamma(S, K, T, r, iv)
                # Long 1 unit of each option at mid (conceptual) — net gamma for equal-weight long strip
                net_gamma += gam
                iv_by_day_strike[tape_day][K].append(iv)
                if q["spread"] is not None:
                    spread_by_day_strike[tape_day][K].append(float(q["spread"]))
                rows_out.append(
                    {
                        "tape_day": tape_day,
                        "timestamp": ts,
                        "strike": K,
                        "S": S,
                        "mid_option": mid,
                        "iv": iv,
                        "delta": dlt,
                        "gamma": gam,
                        "spread": q["spread"],
                        "tte_days": tte_days,
                    }
                )
            if net_gamma > 0:
                gamma_net_by_day[tape_day].append(net_gamma)
            prev_S = S

    # Summaries
    summary: dict = {
        "tte_assumption": (
            "TTE_days = 8 - tape_day_index for historical Round-3 CSV day column "
            "0,1,2 (matches round3description.txt pattern: day1->8d, day2->7d, day3->6d "
            "relative to final sim TTE=5d at start of Round 3)."
        ),
        "mean_iv_by_day_strike": {},
        "mean_spread_by_day_strike": {},
        "mean_net_gamma_equal_long_strip": {},
    }
    for d in iv_by_day_strike:
        summary["mean_iv_by_day_strike"][str(d)] = {
            str(k): round(sum(v) / len(v), 4) for k, v in sorted(iv_by_day_strike[d].items())
        }
    for d in spread_by_day_strike:
        summary["mean_spread_by_day_strike"][str(d)] = {
            str(k): round(sum(v) / len(v), 2) for k, v in sorted(spread_by_day_strike[d].items()) if v
        }
    for d, vals in gamma_net_by_day.items():
        if vals:
            summary["mean_net_gamma_equal_long_strip"][str(d)] = round(sum(vals) / len(vals), 8)

    _OUT.mkdir(parents=True, exist_ok=True)
    detail_csv = _OUT / "iv_greeks_timeseries_sample.csv"
    with detail_csv.open("w", newline="", encoding="utf-8") as f:
        if rows_out:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)

    summary_path = _OUT / "iv_greeks_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"wrote": [str(detail_csv), str(summary_path)], "sample_rows": len(rows_out)}, indent=2))


if __name__ == "__main__":
    main()
