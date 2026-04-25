#!/usr/bin/env python3
"""
Deep tape analysis: extract move propagation to voucher mid + spread behavior.

Outputs:
- family12_outputs/extract_voucher_response.json

Metrics by voucher and day:
- beta1: OLS slope dOption_mid / dExtract_mid (same timestamp step)
- beta_up / beta_down asymmetry by sign(dExtract)
- corr contemporaneous
- lead/lag corr: corr(dOpt_t, dS_{t-1}) and corr(dOpt_t, dS_{t+1})
- spread widening after extract shocks:
    mean spread at baseline,
    mean spread where |dExtract| in top decile,
    ratio shock/base
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ROUND3 = ROOT / "Prosperity4Data" / "ROUND_3"
OUT = Path(__file__).resolve().parent / "family12_outputs"
UNDER = "VELVETFRUIT_EXTRACT"
VOUCHERS = [
    "VEV_4000","VEV_4500","VEV_5000","VEV_5100","VEV_5200",
    "VEV_5300","VEV_5400","VEV_5500","VEV_6000","VEV_6500",
]


def pearson(x: list[float], y: list[float]) -> float | None:
    n = len(x)
    if n < 3 or len(y) != n:
        return None
    mx = sum(x) / n
    my = sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx <= 0 or syy <= 0:
        return None
    sxy = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    return sxy / math.sqrt(sxx * syy)


def slope(x: list[float], y: list[float]) -> float | None:
    n = len(x)
    if n < 3 or len(y) != n:
        return None
    mx = sum(x) / n
    my = sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    if sxx <= 0:
        return None
    sxy = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    return sxy / sxx


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    idx = int(q * (len(arr) - 1))
    return arr[idx]


def load_day(day: int):
    path = ROUND3 / f"prices_round_3_day_{day}.csv"
    by_ts: dict[int, dict[str, dict]] = defaultdict(dict)
    with path.open(encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            ts = int(row["timestamp"])
            prod = row["product"]
            bp = row.get("bid_price_1", "")
            ap = row.get("ask_price_1", "")
            spr = None
            if bp != "" and ap != "":
                spr = int(ap) - int(bp)
            by_ts[ts][prod] = {
                "mid": float(row["mid_price"]),
                "spread": spr,
            }
    return by_ts


def analyze_day(day: int):
    by_ts = load_day(day)
    ts_sorted = sorted(by_ts.keys())
    # build extract series
    s_ts = []
    s_mid = []
    for ts in ts_sorted:
        u = by_ts[ts].get(UNDER)
        if not u:
            continue
        s_ts.append(ts)
        s_mid.append(u["mid"])
    if len(s_mid) < 5:
        return {}
    ds = [s_mid[i] - s_mid[i - 1] for i in range(1, len(s_mid))]
    shock_thr = percentile([abs(x) for x in ds], 0.9)

    out = {}
    for sym in VOUCHERS:
        opt_mid = []
        opt_spread = []
        for ts in s_ts:
            r = by_ts[ts].get(sym)
            if not r:
                opt_mid.append(None)
                opt_spread.append(None)
            else:
                opt_mid.append(r["mid"])
                opt_spread.append(r["spread"])

        dS, dO, spr_base, spr_shock = [], [], [], []
        for i in range(1, len(s_mid)):
            if opt_mid[i] is None or opt_mid[i - 1] is None:
                continue
            dsi = s_mid[i] - s_mid[i - 1]
            doi = opt_mid[i] - opt_mid[i - 1]
            dS.append(dsi)
            dO.append(doi)
            sp = opt_spread[i]
            if sp is not None:
                spr_base.append(float(sp))
                if abs(dsi) >= shock_thr:
                    spr_shock.append(float(sp))

        # asymmetry
        up_idx = [i for i, x in enumerate(dS) if x > 0]
        dn_idx = [i for i, x in enumerate(dS) if x < 0]
        dS_up = [dS[i] for i in up_idx]
        dO_up = [dO[i] for i in up_idx]
        dS_dn = [dS[i] for i in dn_idx]
        dO_dn = [dO[i] for i in dn_idx]

        # lead/lag correlations
        # corr(dO_t, dS_{t-1}) and corr(dO_t, dS_{t+1})
        lead = None
        lag = None
        if len(dS) > 3:
            lag = pearson(dO[1:], dS[:-1])
            lead = pearson(dO[:-1], dS[1:])

        mean_base = (sum(spr_base) / len(spr_base)) if spr_base else None
        mean_shock = (sum(spr_shock) / len(spr_shock)) if spr_shock else None
        shock_ratio = (mean_shock / mean_base) if (mean_base and mean_shock is not None) else None

        out[sym] = {
            "n": len(dS),
            "beta1": slope(dS, dO),
            "corr": pearson(dS, dO),
            "beta_up": slope(dS_up, dO_up),
            "beta_down": slope(dS_dn, dO_dn),
            "corr_dO_t_dS_t_minus_1": lag,
            "corr_dO_t_dS_t_plus_1": lead,
            "spread_mean": mean_base,
            "spread_mean_shock": mean_shock,
            "spread_shock_ratio": shock_ratio,
            "shock_threshold_abs_dS": shock_thr,
        }
    return out


def main():
    payload = {"method": "ticker-level response and spread metrics from ROUND_3 prices tape", "by_day": {}}
    for d in (0, 1, 2):
        payload["by_day"][str(d)] = analyze_day(d)

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "extract_voucher_response.json"
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(p)


if __name__ == "__main__":
    main()
