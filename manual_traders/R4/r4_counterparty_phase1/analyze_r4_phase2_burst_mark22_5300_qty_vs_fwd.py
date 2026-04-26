#!/usr/bin/env python3
"""Phase 2.7-style inventory proxy: at joint-tight + M01→M22 burst timestamps, relate
**Mark 22 sell quantity on VEV_5300** in that print batch to subsequent **VEV_5300**
and **VELVETFRUIT_EXTRACT** fwd20 mids (K=20 price rows).

Not full MM inventory — population-level pressure from Mark 22 on the wing at
orchestration timestamps.
"""
from __future__ import annotations

import csv
import json
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "r4_phase2_burst_m22_5300_qty_vs_fwd.json"
DAYS = (1, 2, 3)
TH = 2.0
BURST_MIN = 4
K = 20
S5200, S5300 = "VEV_5200", "VEV_5300"
EX = "VELVETFRUIT_EXTRACT"


def load_burst_rows_by_ts():
    by: dict[tuple[int, int], list[tuple]] = defaultdict(list)
    for d in DAYS:
        p = DATA / f"trades_round_4_day_{d}.csv"
        with open(p, newline="") as f:
            r = csv.DictReader(f, delimiter=";")
            for row in r:
                by[(d, int(row["timestamp"]))].append(
                    (
                        str(row["symbol"]),
                        str(row["buyer"]).strip(),
                        str(row["seller"]).strip(),
                        int(float(row["quantity"])),
                    )
                )
    return by


def burst_m01_m22(rows: list[tuple]) -> bool:
    if len(rows) < BURST_MIN:
        return False
    return any(b == "Mark 01" and s == "Mark 22" for _sym, b, s, _q in rows)


def m22_sell_qty_5300(rows: list[tuple]) -> int:
    return sum(q for sym, _b, s, q in rows if sym == S5300 and s == "Mark 22")


def fwd_series(tss: list[int], mids: list[float], ts: int) -> float | None:
    i = bisect_right(tss, ts) - 1
    if i < 0:
        i = 0
    j = i + K
    if j >= len(mids):
        return None
    return mids[j] - mids[i]


def main() -> None:
    tr = load_burst_rows_by_ts()
    rows_data: list[dict] = []

    for day in DAYS:
        sp52: dict[int, float] = {}
        sp53: dict[int, float] = {}
        tss53: list[int] = []
        mid53: list[float] = []
        tss_ex: list[int] = []
        mid_ex: list[float] = []
        path = DATA / f"prices_round_4_day_{day}.csv"
        with open(path, newline="") as f:
            r = csv.DictReader(f, delimiter=";")
            for row in r:
                if int(row["day"]) != day:
                    continue
                sym = row["product"]
                ts = int(row["timestamp"])
                bid = float(row["bid_price_1"])
                ask = float(row["ask_price_1"])
                sp = ask - bid if ask > bid else 0.0
                mp = float(row["mid_price"])
                if sym == S5200:
                    sp52[ts] = sp
                elif sym == S5300:
                    sp53[ts] = sp
                    tss53.append(ts)
                    mid53.append(mp)
                elif sym == EX:
                    tss_ex.append(ts)
                    mid_ex.append(mp)

        if len(tss53) < K + 2 or len(tss_ex) < K + 2:
            continue
        for ts in sorted(sp53.keys()):
            if ts not in sp52:
                continue
            if sp52[ts] > TH or sp53[ts] > TH:
                continue
            trows = tr.get((day, ts), [])
            if not burst_m01_m22(trows):
                continue
            q22 = m22_sell_qty_5300(trows)
            f53 = fwd_series(tss53, mid53, ts)
            fex = fwd_series(tss_ex, mid_ex, ts)
            if f53 is None or fex is None:
                continue
            rows_data.append(
                {
                    "day": day,
                    "ts": ts,
                    "m22_sell_qty_5300": q22,
                    "fwd20_5300": f53,
                    "fwd20_extract": fex,
                }
            )

    if not rows_data:
        OUT.write_text(json.dumps({"error": "no rows"}, indent=2))
        return

    qs = sorted(r["m22_sell_qty_5300"] for r in rows_data)
    n = len(qs)
    lo = qs[n // 4]
    hi = qs[(3 * n) // 4]

    def stat(xs: list[float]) -> dict:
        if not xs:
            return {"n": 0, "mean": None}
        return {"n": len(xs), "mean": round(sum(xs) / len(xs), 6)}

    low = [r for r in rows_data if r["m22_sell_qty_5300"] <= lo]
    high = [r for r in rows_data if r["m22_sell_qty_5300"] >= hi]

    out = {
        "K": K,
        "TH": TH,
        "BURST_MIN": BURST_MIN,
        "n_events": len(rows_data),
        "quartiles_m22_sell_qty_5300": {"p25": lo, "p75": hi},
        "low_m22_pressure": {
            "n": len(low),
            "mean_fwd20_5300": stat([r["fwd20_5300"] for r in low])["mean"],
            "mean_fwd20_extract": stat([r["fwd20_extract"] for r in low])["mean"],
        },
        "high_m22_pressure": {
            "n": len(high),
            "mean_fwd20_5300": stat([r["fwd20_5300"] for r in high])["mean"],
            "mean_fwd20_extract": stat([r["fwd20_extract"] for r in high])["mean"],
        },
        "pooled_correlation_note": "Compare high vs low Mark22 sell qty on 5300 within same burst regime; if high pressure associates with lower fwd20, supports fade / skew-away-from-M22-sell inventory stories.",
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
