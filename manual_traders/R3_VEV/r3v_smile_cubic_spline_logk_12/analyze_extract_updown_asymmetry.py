#!/usr/bin/env python3
"""Per-strike asymmetry: beta of dVoucher on dExtract when dExtract>0 vs dExtract<0.
Also mean spread on up vs down extract steps. ROUND_3 tapes, TTE implicit in day index only for context.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "Prosperity4Data" / "ROUND_3"
OUT = Path(__file__).resolve().parent / "analysis_outputs"
EX = "VELVETFRUIT_EXTRACT"
VEVS = [
    "VEV_4000", "VEV_4500", "VEV_5000", "VEV_5100", "VEV_5200",
    "VEV_5300", "VEV_5400", "VEV_5500", "VEV_6000", "VEV_6500",
]


def f(x: str):
    if x == "":
        return None
    try:
        return float(x)
    except ValueError:
        return None


def beta(x, y):
    if len(x) < 10:
        return None
    xx = np.array(x)
    yy = np.array(y)
    vx = float(np.var(xx))
    if vx <= 1e-14:
        return None
    return float(np.cov(xx, yy, ddof=0)[0, 1] / vx)


def mean_spread(rows, ts, sym):
    s = 0.0
    n = 0
    for t in ts:
        r = rows[t].get(sym)
        if not r:
            continue
        b, a = f(r.get("bid_price_1", "")), f(r.get("ask_price_1", ""))
        if b is None or a is None:
            continue
        s += a - b
        n += 1
    return s / n if n else None


def run_day(day: int):
    path = DATA / f"prices_round_3_day_{day}.csv"
    rows_list = list(csv.DictReader(path.open(), delimiter=";"))
    by_ts: dict[int, dict[str, dict]] = defaultdict(dict)
    for r in rows_list:
        by_ts[int(r["timestamp"])][r["product"]] = r
    ts_sorted = sorted(by_ts.keys())

    ex_m = []
    for t in ts_sorted:
        er = by_ts[t].get(EX)
        ex_m.append(f(er["mid_price"]) if er else None)

    out = []
    for sym in VEVS:
        xup, yup, xdn, ydn = [], [], [], []
        ts_up, ts_dn = [], []
        for i in range(1, len(ts_sorted)):
            if ex_m[i] is None or ex_m[i - 1] is None:
                continue
            dx = ex_m[i] - ex_m[i - 1]
            vr = by_ts[ts_sorted[i]].get(sym)
            if not vr:
                continue
            vm0 = f(vr["mid_price"])
            v0 = by_ts[ts_sorted[i - 1]].get(sym)
            if not v0:
                continue
            vm1 = f(v0["mid_price"])
            if vm0 is None or vm1 is None:
                continue
            dv = vm0 - vm1
            if dx > 0.5:
                xup.append(dx)
                yup.append(dv)
                ts_up.append(ts_sorted[i])
            elif dx < -0.5:
                xdn.append(dx)
                ydn.append(dv)
                ts_dn.append(ts_sorted[i])

        bu = beta(xup, yup)
        bd = beta(xdn, ydn)
        msup = mean_spread(by_ts, ts_up, sym)
        msdn = mean_spread(by_ts, ts_dn, sym)
        msall = mean_spread(by_ts, ts_sorted, sym)
        out.append(
            {
                "day": day,
                "symbol": sym,
                "n_up": len(xup),
                "n_down": len(xdn),
                "beta_up": bu,
                "beta_down": bd,
                "mean_spread_on_upstep": msup,
                "mean_spread_on_downstep": msdn,
                "mean_spread_all": msall,
            }
        )
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    allr = []
    for d in (0, 1, 2):
        allr.extend(run_day(d))
    p = OUT / "extract_updown_asymmetry.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(allr[0].keys()))
        w.writeheader()
        w.writerows(allr)
    print("wrote", p, "rows", len(allr))


if __name__ == "__main__":
    main()
