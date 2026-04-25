#!/usr/bin/env python3
"""
Tape: mid minus BS(PWL-IV) by strike, using same knot IV construction as the live trader.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from _r3v_smile_core import (  # noqa: E402
    bs_call_delta,
    bs_call_price,
    implied_vol_bisect,
    pwl_iv_strike,
    t_years_effective,
)

U = "VELVETFRUIT_EXTRACT"
# IV + knot construction uses the same STRIKES band as trader (inner + wings for IV surface)
ALL_K = (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)
ANALYZE = (5000, 5100, 5200, 5300, 5400, 5500)
KNOTS = (5000, 5200, 5400)
DATA = Path("Prosperity4Data/ROUND_3")
OUT = ROOT / "analysis_outputs" / "tape_pwl_misfit_by_strike.json"


def _median(xs: list[float]) -> float | None:
    ys = [x for x in xs if x is not None and math.isfinite(x)]
    if not ys:
        return None
    ys.sort()
    return ys[len(ys) // 2]


def knot_ivs_from_surface(strike_iv: dict[int, float | None]) -> tuple[float, float, float] | None:
    def ivs(ks: tuple[int, ...]) -> list[float]:
        out: list[float] = []
        for k in ks:
            v = strike_iv.get(k)
            if v is not None and math.isfinite(v) and v > 0:
                out.append(float(v))
        return out

    m0 = _median(ivs((5000, 5100)))
    m1 = _median(ivs((5100, 5200, 5300)))
    m2 = _median(ivs((5300, 5400, 5500)))
    if m0 is None or m1 is None or m2 is None:
        return None

    def clip(x: float) -> float:
        return max(0.04, min(3.5, x))

    return (clip(m0), clip(m1), clip(m2))


def qtile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    i = int(q * (len(xs) - 1))
    return float(xs[i])


def main() -> None:
    from collections import OrderedDict

    by_day: dict = OrderedDict()
    for d in (0, 1, 2):
        p = DATA / f"prices_round_3_day_{d}.csv"
        rows: list[dict[str, str]] = []
        with open(p, newline="") as f:
            for r in csv.DictReader(f, delimiter=";"):
                rows.append(r)

        by_ts: dict[int, dict[str, float]] = defaultdict(dict)
        for r in rows:
            ts = int(r["timestamp"])
            prod = r["product"]
            by_ts[ts][prod] = float(r["mid_price"])

        misfit: dict[str, list[float]] = {f"VEV_{k}": [] for k in ANALYZE}
        delta_h: dict[str, list[float]] = {f"VEV_{k}": [] for k in ANALYZE}
        spread_h: dict[str, list[float]] = {f"VEV_{k}": [] for k in ANALYZE}

        for r in rows:
            prod = r["product"]
            if not prod.startswith("VEV_") or prod not in misfit:
                continue
            try:
                bb = int(float(r["bid_price_1"]))
                ba = int(float(r["ask_price_1"]))
                spread_h[prod].append(float(ba - bb))
            except (ValueError, TypeError, KeyError):
                pass

        for ts in sorted(by_ts.keys()):
            b = by_ts[ts]
            if U not in b:
                continue
            s_mid = b[U]
            strike_mids: dict[int, float] = {}
            for k in ALL_K:
                sym = f"VEV_{k}"
                if sym in b:
                    strike_mids[k] = b[sym]
            if len([k for k in (5000, 5100, 5200, 5300, 5400, 5500) if k in strike_mids]) < 4:
                continue
            T = t_years_effective(d, ts)
            strike_iv: dict[int, float | None] = {}
            for K in ALL_K:
                mid = strike_mids.get(K)
                if mid is None:
                    strike_iv[K] = None
                    continue
                strike_iv[K] = implied_vol_bisect(mid, s_mid, float(K), T)

            kivs = knot_ivs_from_surface(strike_iv)
            if kivs is None:
                continue

            for k in ANALYZE:
                sym = f"VEV_{k}"
                if sym not in b:
                    continue
                mid = b[sym]
                sig = pwl_iv_strike(float(k), KNOTS, kivs)
                if not math.isfinite(sig) or sig <= 0:
                    continue
                theo = bs_call_price(s_mid, float(k), T, sig)
                if not math.isfinite(theo):
                    continue
                misfit[sym].append(float(mid) - theo)
                delta_h[sym].append(
                    float(bs_call_delta(s_mid, float(k), T, sig))
                )

        per_sym: dict = {}
        for k in ANALYZE:
            sym = f"VEV_{k}"
            xs = misfit[sym]
            if not xs:
                per_sym[sym] = {"n": 0}
                continue
            per_sym[sym] = {
                "n": len(xs),
                "misfit_median": float(statistics.median(xs)),
                "misfit_p10": qtile(xs, 0.1),
                "misfit_p90": qtile(xs, 0.9),
                "abs_median": float(statistics.median([abs(x) for x in xs])),
                "spread_median": float(statistics.median(spread_h[sym]))
                if spread_h[sym]
                else None,
                "delta_median": float(statistics.median(delta_h[sym])),
            }
        by_day[str(d)] = per_sym

    obj = {
        "method": "Per ts: S=VELVET mid; IV from bisection; knots from 5000-5500 medians like trader; misfit=mid-BS(PWL). T=t_years_effective(day,ts).",
        "by_day": by_day,
    }
    OUT.write_text(json.dumps(obj, indent=2))
    print(OUT)
    for d in ("0", "1", "2"):
        for sym in ("VEV_5200", "VEV_5300", "VEV_5100"):
            m = by_day[d].get(sym, {})
            if m.get("n"):
                print(
                    d,
                    sym,
                    "med_misfit",
                    round(m["misfit_median"], 2),
                    "abs",
                    round(m["abs_median"], 2),
                )


if __name__ == "__main__":
    main()
