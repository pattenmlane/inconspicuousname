#!/usr/bin/env python3
"""Round 4 Phase 3 (inclineGod-style): spread aggregates vs forward mids on the
joint inner-join panel (VEV_5200, VEV_5300, VELVETFRUIT_EXTRACT).

Convention aligned with round3work/vouchers_final_strategy/analyze_vev_5200_5300_tight_gate_r3.py:
inner join on timestamp; L1 spread = ask1-bid1; Sonic tight = s5200<=TH & s5300<=TH.

JSON: correlations on **tight** rows (spread_sum, extract spread vs fwd); **pair grid**
(integer s5200×s5300) mean fwd because spread_sum is often 4 when both legs tight.
K=20 row-forward on each mid series.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "r4_phase3_spread_vs_fwd_panels.json"
DAYS = (1, 2, 3)
TH = 2.0
K = 20
S5200, S5300 = "VEV_5200", "VEV_5300"
EX = "VELVETFRUIT_EXTRACT"


def _float(x: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load_one(day: int, product: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    path = DATA / f"prices_round_4_day_{day}.csv"
    with open(path, newline="") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            if int(row["day"]) != day or row["product"] != product:
                continue
            ts = int(row["timestamp"])
            bid = _float(row["bid_price_1"])
            ask = _float(row["ask_price_1"])
            mid = _float(row["mid_price"])
            if math.isnan(mid):
                continue
            if math.isnan(bid) or math.isnan(ask):
                bid, ask = mid, mid
            sp = ask - bid if ask > bid else 0.0
            out[ts] = {"spr": sp, "mid": mid}
    return out


def pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3 or len(ys) != n:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx < 1e-12 or dy < 1e-12:
        return None
    return num / (dx * dy)


def main() -> None:
    pooled_ss: list[float] = []
    pooled_sext: list[float] = []
    pooled_fex: list[float] = []
    pooled_f53: list[float] = []
    pooled_afex: list[float] = []
    per_day_out: dict[str, dict] = {}

    for day in DAYS:
        a = load_one(day, S5200)
        b = load_one(day, S5300)
        e = load_one(day, EX)
        tss = sorted(set(a) & set(b) & set(e))
        mids_ex = [e[ts]["mid"] for ts in tss]
        mids_53 = [b[ts]["mid"] for ts in tss]
        spr5200 = [a[ts]["spr"] for ts in tss]
        spr5300 = [b[ts]["spr"] for ts in tss]
        spr_ext = [e[ts]["spr"] for ts in tss]
        n = len(tss)
        tight_ss: list[float] = []
        tight_sext: list[float] = []
        tight_fex: list[float] = []
        tight_f53: list[float] = []
        tight_afex: list[float] = []
        grid: dict[tuple[int, int], list[tuple[float, float]]] = {}
        for i in range(n - K):
            s5200, s5300 = spr5200[i], spr5300[i]
            if s5200 > TH or s5300 > TH:
                continue
            fex = mids_ex[i + K] - mids_ex[i]
            f53 = mids_53[i + K] - mids_53[i]
            ss = s5200 + s5300
            sx = spr_ext[i]
            tight_ss.append(ss)
            tight_sext.append(sx)
            tight_fex.append(fex)
            tight_f53.append(f53)
            tight_afex.append(abs(fex))
            pooled_ss.append(ss)
            pooled_sext.append(sx)
            pooled_fex.append(fex)
            pooled_f53.append(f53)
            pooled_afex.append(abs(fex))
            gk = (int(round(s5200)), int(round(s5300)))
            grid.setdefault(gk, []).append((fex, f53))

        grid_stats = {}
        for (i5200, i5300), pairs in sorted(grid.items()):
            if not pairs:
                continue
            fe = [p[0] for p in pairs]
            f5 = [p[1] for p in pairs]
            grid_stats[f"s5200={i5200}_s5300={i5300}"] = {
                "n": len(pairs),
                "mean_fwd_ex": round(sum(fe) / len(fe), 6),
                "mean_fwd_5300": round(sum(f5) / len(f5), 6),
            }

        per_day_out[str(day)] = {
            "n_inner_join": n,
            "n_tight_rows_with_fwd": len(tight_ss),
            "corr_spread_sum_fwd_ex_tight": pearson(tight_ss, tight_fex),
            "corr_spread_sum_abs_fwd_ex_tight": pearson(tight_ss, tight_afex),
            "corr_spread_sum_fwd_5300_tight": pearson(tight_ss, tight_f53),
            "corr_extract_spread_fwd_ex_tight": pearson(tight_sext, tight_fex),
            "corr_extract_spread_abs_fwd_ex_tight": pearson(tight_sext, tight_afex),
            "mean_fwd_ex_tight": (
                round(sum(tight_fex) / len(tight_fex), 6) if tight_fex else None
            ),
            "mean_fwd_5300_tight": (
                round(sum(tight_f53) / len(tight_f53), 6) if tight_f53 else None
            ),
            "tight_spread_pair_grid_mean_fwd": grid_stats,
        }

    out = {
        "TH": TH,
        "K": K,
        "definition": "inner_join timestamps 5200+5300+extract; fwd = mid[i+K]-mid[i]; tight = s5200<=TH & s5300<=TH. spread_sum often ==4 when both max-tight — use pair grid.",
        "per_day": per_day_out,
        "pooled_tight_rows": {
            "n": len(pooled_ss),
            "corr_spread_sum_fwd_ex": pearson(pooled_ss, pooled_fex),
            "corr_spread_sum_abs_fwd_ex": pearson(pooled_ss, pooled_afex),
            "corr_spread_sum_fwd_5300": pearson(pooled_ss, pooled_f53),
            "corr_extract_spread_fwd_ex": pearson(pooled_sext, pooled_fex),
            "corr_extract_spread_abs_fwd_ex": pearson(pooled_sext, pooled_afex),
        },
        "interpretation": "Within Sonic-tight, s5200+s5300 is nearly constant at 4 — tertiles collapse; **pair grid** (integer L1 spreads) separates mean fwd. Pooled corr spread_sum vs fwd is ~0 / slightly negative on this tape; extract spread vs fwd per day in JSON.",
    }
    OUT.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
