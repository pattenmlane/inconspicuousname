#!/usr/bin/env python3
"""Lead/lag dU vs dOption; spread in high |dU| vs rest. ROUND_3 days 0-2."""
from __future__ import annotations
import csv
import json
from collections import defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parents[3] / "Prosperity4Data" / "ROUND_3"
OUT = Path(__file__).resolve().parent / "analysis_lead_lag_spread_ex.json"
SAMPLE = ["VEV_5000", "VEV_5200", "VEV_4000", "VEV_6500"]
UNDER = "VELVETFRUIT_EXTRACT"
SHOCK = 3.0


def load(path: Path):
    d: dict[int, dict[str, dict]] = defaultdict(dict)
    with path.open() as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            d[int(row["timestamp"])][row["product"]] = row
    return d


def m(row) -> float | None:
    try:
        return float(row["mid_price"])
    except (KeyError, ValueError, TypeError):
        return None


def spr(row) -> float | None:
    try:
        return float(row["ask_price_1"]) - float(row["bid_price_1"])
    except (KeyError, ValueError, TypeError):
        return None


def corr(x: list, y: list) -> float | None:
    n = len(x)
    if n < 10 or n != len(y):
        return None
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dxe = sum((a - mx) ** 2 for a in x) ** 0.5
    dye = sum((b - my) ** 2 for b in y) ** 0.5
    if dxe < 1e-12 or dye < 1e-12:
        return None
    return num / (dxe * dye)


def main():
    out = {
        "method": "Consecutive-timestamp first differences. Same-step: dU_t, dO_t. Lag+1: dU_t, dO_{t+1}. Spreads: bucket by |dU| at end time of step.",
        "shock_abs_dU": SHOCK,
        "per_symbol": {},
    }
    for sym in SAMPLE:
        dus: list[float] = []
        dos0: list[float] = []
        dus1: list[float] = []
        dos1: list[float] = []
        sh_hi: list[float] = []
        sh_lo: list[float] = []
        n_hi = 0
        n_lo = 0
        for day in (0, 1, 2):
            by = load(DATA / f"prices_round_3_day_{day}.csv")
            ts = sorted(by.keys())
            pr_u = None
            pr_o: float | None = None
            buf: list[tuple[float, float, float | None]] = []
            for t in ts:
                if UNDER not in by[t] or sym not in by[t]:
                    continue
                u = m(by[t][UNDER])
                o = m(by[t][sym])
                if u is None or o is None:
                    continue
                if pr_u is not None:
                    d_u = u - pr_u
                    d_o = o - pr_o
                    sp = spr(by[t][sym])
                    buf.append((d_u, d_o, sp))
                pr_u, pr_o = u, o
            for j in range(len(buf) - 1):
                du, do0, _ = buf[j]
                _, do_next, _ = buf[j + 1]
                dus.append(du)
                dos0.append(do0)
                dus1.append(du)
                dos1.append(do_next)
            for du, do0, sp in buf:
                if sp is not None:
                    if abs(du) >= SHOCK:
                        sh_hi.append(sp)
                        n_hi += 1
                    else:
                        sh_lo.append(sp)
                        n_lo += 1
        c0 = corr(dus, dos0) if dus else None
        c1 = corr(dus1, dos1) if dus1 else None
        mhi = sum(sh_hi) / len(sh_hi) if sh_hi else None
        mlo = sum(sh_lo) / len(sh_lo) if sh_lo else None
        out["per_symbol"][sym] = {
            "n_d_pairs": len(dus),
            "n_lag_pairs": len(dus1),
            "corr_dU_dO_same": c0,
            "corr_dU_t_dO_tplus1": c1,
            "mean_spread_after_high_absdU": mhi,
            "n_spread_high": n_hi,
            "mean_spread_after_low_absdU": mlo,
            "n_spread_low": n_lo,
        }
    OUT.write_text(json.dumps(out, indent=2))
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
