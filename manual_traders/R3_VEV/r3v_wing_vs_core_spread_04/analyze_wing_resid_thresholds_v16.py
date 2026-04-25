"""
Round-3: fraction of subsampled timestamps where any wing strike (WING_KS in v15)
has |IV resid| above a grid of thresholds. IV = BS inversion (r=0), quad IV(m_t).
TTE: csv day 0/1/2 open 8/7/6d + intraday wind per established mapping.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_3"
OUT = Path(__file__).resolve().parent / "wing_resid_thresholds_v16.json"

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOU = [f"VEV_{k}" for k in STRIKES]
U = "VELVETFRUIT_EXTRACT"
WING_KS = {4000, 4500, 5400, 5500, 6000, 6500}
MIN_TV = 0.045
STEP = 50
THRS = [0.008, 0.010, 0.012, 0.014, 0.016]


def t_years(day: int, ts: int) -> float:
    dte = max(8.0 - float(day) - (int(ts) // 100) / 10_000.0, 1e-6)
    return dte / 365.0


def bs(S: float, K: float, T: float, s: float) -> float:
    if T <= 0 or s <= 1e-12:
        return max(S - K, 0.0)
    v = s * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * s * s * T) / v
    d2 = d1 - v
    return S * norm.cdf(d1) - K * norm.cdf(d2)


def iv_mid(m: float, S: float, K: float, T: float) -> float | None:
    if m <= max(S - K, 0) + 1e-6 or m >= S - 1e-6 or S <= 0:
        return None

    def f(x: float) -> float:
        return bs(S, K, T, x) - m

    if f(1e-4) > 0 or f(12.0) < 0:
        return None
    return float(brentq(f, 1e-4, 12.0))


def main() -> None:
    counts: dict[str, int] = {f"abs_resid_ge_{t}": 0 for t in THRS}
    n_steps = 0
    for day in (0, 1, 2):
        df = pd.read_csv(DATA / f"prices_round_3_day_{day}.csv", sep=";")
        p = df.pivot_table(index="timestamp", columns="product", values="mid_price", aggfunc="first")
        p = p[[U] + [c for c in VOU if c in p.columns]].sort_index()
        for ts in p.index[::STEP]:
            row = p.loc[ts]
            S = float(row[U])
            T = t_years(day, int(ts))
            if S <= 0 or T <= 0:
                continue
            srt = math.sqrt(T)
            xs, ys = [], []
            for v in VOU:
                if v not in row.index or pd.isna(row[v]):
                    continue
                m = float(row[v])
                K = int(v.split("_")[1])
                intr = max(S - K, 0.0)
                if m <= 0 or (m - intr) / m < MIN_TV:
                    continue
                sig = iv_mid(m, S, K, T)
                if sig is None:
                    continue
                xs.append(math.log(K / S) / srt)
                ys.append(sig)
            if len(xs) < 6:
                continue
            c = np.polyfit(np.asarray(xs), np.asarray(ys), 2)
            a, b, c0 = float(c[0]), float(c[1]), float(c[2])

            def hat(m: float) -> float:
                return a * m * m + b * m + c0

            max_abs = 0.0
            for k in WING_KS:
                v = f"VEV_{k}"
                if v not in row.index or pd.isna(row[v]):
                    continue
                m = float(row[v])
                intr = max(S - k, 0.0)
                if m <= 0 or (m - intr) / m < MIN_TV:
                    continue
                sig = iv_mid(m, S, k, T)
                if sig is None:
                    continue
                m_t = math.log(k / S) / srt
                r = sig - hat(m_t)
                max_abs = max(max_abs, abs(r))
            n_steps += 1
            for t in THRS:
                if max_abs >= t:
                    counts[f"abs_resid_ge_{t}"] += 1

    frac = {k: counts[k] / n_steps if n_steps else 0.0 for k in counts}
    OUT.write_text(
        json.dumps(
            {
                "n_subsampled_fits": n_steps,
                "step": STEP,
                "wing_strikes": sorted(WING_KS),
                "fraction_timestamps_max_abs_wing_resid_ge_threshold": {str(t): frac[f"abs_resid_ge_{t}"] for t in THRS},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(OUT.read_text())


if __name__ == "__main__":
    main()
