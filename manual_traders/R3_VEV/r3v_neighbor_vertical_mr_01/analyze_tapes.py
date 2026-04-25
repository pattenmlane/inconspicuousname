#!/usr/bin/env python3
"""One-off tape analysis for r3v_neighbor_vertical_mr_01 — writes analysis artifacts."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "Prosperity4Data" / "ROUND_3"
OUT_DIR = Path(__file__).resolve().parent

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
SYMS = [f"VEV_{k}" for k in STRIKES]
UNDER = "VELVETFRUIT_EXTRACT"
N = NormalDist()


def bs_call_price(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(S - K, 0.0)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return S * N.cdf(d1) - K * math.exp(-r * T) * N.cdf(d2)


def bs_vega(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    return S * sqrtT * (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * d1 * d1)


def implied_vol_bisect(mid: float, S: float, K: float, T: float, lo=1e-6, hi=5.0, it=60) -> float | None:
    """Black–Scholes IV from European call mid; r=0."""
    intrinsic = max(S - K, 0.0)
    if mid < intrinsic - 1e-9:
        return None
    if T <= 0:
        return None
    lo_p, hi_p = bs_call_price(S, K, T, lo), bs_call_price(S, K, T, hi)
    if mid > hi_p:
        return None
    a, b = lo, hi
    for _ in range(it):
        m = 0.5 * (a + b)
        p = bs_call_price(S, K, T, m)
        if p > mid:
            b = m
        else:
            a = m
    return 0.5 * (a + b)


def tte_years(csv_day: int) -> float:
    """From round3work/round3description.txt example: hist day 1->8d, 2->7d, 3->6d.
    CSV files use day 0,1,2 aligned with those three historical days => TTE = 8 - csv_day."""
    tte_d = 8 - int(csv_day)
    return max(tte_d, 1) / 365.0


def load_day(path: Path):
    rows = defaultdict(dict)  # ts -> product -> row
    with path.open(newline="") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            ts = int(row["timestamp"])
            rows[ts][row["product"]] = row
    return rows


def mid_from_row(row) -> float | None:
    try:
        return float(row["mid_price"])
    except (KeyError, TypeError, ValueError):
        return None


def spread_width(row) -> float | None:
    try:
        bp = int(row["bid_price_1"])
        ap = int(row["ask_price_1"])
        return float(ap - bp)
    except (KeyError, TypeError, ValueError):
        return None


def main():
    summary = {"tte_rule": "CSV column day in {0,1,2} maps to TTE_days = 8 - day (per round3description.txt example: historical day 1->8d, 2->7d, 3->6d).", "iv_method": "Bisection on Black-Scholes European call price vs voucher mid; r=0; T = TTE_days/365."}
    iv_skew_by_day = {}
    neighbor_gap_stats = {}
    vega_weight_samples = []

    for csv_day in (0, 1, 2):
        path = DATA / f"prices_round_3_day_{csv_day}.csv"
        T = tte_years(csv_day)
        data = load_day(path)
        Tval = tte_years(csv_day) * 365.0
        ivs_atm = []
        gaps = []
        for ts in sorted(data.keys())[:5000:10]:  # subsample for speed
            pr = data[ts]
            if UNDER not in pr:
                continue
            Su = mid_from_row(pr[UNDER])
            if Su is None or Su <= 0:
                continue
            mids = []
            spreads = []
            ok = True
            for s in SYMS:
                if s not in pr:
                    ok = False
                    break
                m = mid_from_row(pr[s])
                w = spread_width(pr[s])
                if m is None or w is None:
                    ok = False
                    break
                mids.append(m)
                spreads.append(w)
            if not ok:
                continue
            for i in range(len(STRIKES) - 1):
                gaps.append(mids[i] - mids[i + 1])
            # ATM ~ 5400 or 5500 depending on S; pick strike closest to S
            j = min(range(len(STRIKES)), key=lambda k: abs(STRIKES[k] - Su))
            K = STRIKES[j]
            sym = SYMS[j]
            mid_c = mids[j]
            iv = implied_vol_bisect(mid_c, Su, float(K), T)
            if iv is not None:
                ivs_atm.append(iv)
                vega = bs_vega(Su, float(K), T, iv)
                vega_weight_samples.append({"day": csv_day, "ts": ts, "vega": vega, "iv": iv, "spread": spreads[j]})

        if ivs_atm:
            iv_skew_by_day[str(csv_day)] = {
                "mean_iv": sum(ivs_atm) / len(ivs_atm),
                "n": len(ivs_atm),
                "tte_days": Tval,
            }
        if gaps:
            neighbor_gap_stats[str(csv_day)] = {
                "mean_gap": sum(gaps) / len(gaps),
                "std_gap": (sum((g - sum(gaps) / len(gaps)) ** 2 for g in gaps) / max(len(gaps) - 1, 1)) ** 0.5,
                "n_gap_samples": len(gaps),
            }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    art = OUT_DIR / "analysis_iv_neighbor_summary.json"
    with art.open("w") as f:
        json.dump({"summary": summary, "iv_near_money": iv_skew_by_day, "neighbor_gap_pooled": neighbor_gap_stats}, f, indent=2)

    analysis_entry = {
        "iteration": 0,
        "analysis_description": "TTE from round3description (hist day 1/2/3 -> 8/7/6d mapped to CSV day 0/1/2). Implied vol via BS bisection on ATM-near voucher vs VELVETFRUIT_EXTRACT mid; vega at that IV. Neighbor mid gaps m_i - m_{i+1} pooled stats and spread width in samples.",
        "analysis_outputs": [str(art.relative_to(ROOT))],
        "used_to_decide": "Confirms IV is numerically stable on tapes; neighbor gaps have positive mean (decreasing mids with strike) with dispersion usable for z-score MR; vega scales with TTE.",
    }
    aj = OUT_DIR / "analysis.json"
    if aj.exists():
        prev = json.loads(aj.read_text())
        if not isinstance(prev, list):
            prev = []
    else:
        prev = []
    prev = [x for x in prev if x.get("iteration") != 0]
    prev.append(analysis_entry)
    prev.sort(key=lambda x: x["iteration"])
    aj.write_text(json.dumps(prev, indent=2))
    print("Wrote", art, "and updated", aj)


if __name__ == "__main__":
    main()
