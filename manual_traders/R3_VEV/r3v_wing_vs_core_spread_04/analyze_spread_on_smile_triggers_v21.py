"""
When quadratic IV(m_t) would trigger v20 rules on wing strikes, record bid-ask width.
RICH=0.010, CHEAP=-0.022, WING_KS, MIN_TV=0.045; BS IV r=0; T from csv day + intraday wind.
Reads Prosperity4Data/ROUND_3; writes spread_on_smile_triggers_v21.json
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
OUT = Path(__file__).resolve().parent / "spread_on_smile_triggers_v21.json"

RICH, CHEAP = 0.010, -0.022
MIN_TV = 0.045
STEP = 30
WING = {4000, 4500, 5400, 5500, 6000, 6500}
ST = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOU = [f"VEV_{k}" for k in ST]
U = "VELVETFRUIT_EXTRACT"


def t_y(d: int, ts: int) -> float:
    return max(8.0 - float(d) - (int(ts) // 100) / 10_000.0, 1e-6) / 365.0


def bs(S: float, K: float, T: float, s: float) -> float:
    if T <= 0 or s <= 1e-12:
        return max(S - K, 0.0)
    v = s * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * s * s * T) / v
    d2 = d1 - v
    return S * norm.cdf(d1) - K * norm.cdf(d2)


def iv(m: float, S: float, K: float, T: float) -> float | None:
    if m <= max(S - K, 0) + 1e-6 or m >= S - 1e-6 or S <= 0:
        return None

    def f(x: float) -> float:
        return bs(S, K, T, x) - m

    if f(1e-4) > 0 or f(12) < 0:
        return None
    return float(brentq(f, 1e-4, 12.0))


def spread_raw(df: pd.DataFrame, day: int, ts: int, prod: str) -> int | None:
    r = df[(df["day"] == day) & (df["timestamp"] == ts) & (df["product"] == prod)]
    if r.empty:
        return None
    row = r.iloc[0]
    bps = [row[f"bid_price_{i}"] for i in (1, 2, 3) if pd.notna(row.get(f"bid_price_{i}"))]
    aps = [row[f"ask_price_{i}"] for i in (1, 2, 3) if pd.notna(row.get(f"ask_price_{i}"))]
    if not bps or not aps:
        return None
    return int(min(aps) - max(bps))


def main() -> None:
    w_rich, w_cheap, n = [], [], 0
    for day in (0, 1, 2):
        df0 = pd.read_csv(DATA / f"prices_round_3_day_{day}.csv", sep=";")
        p = df0.pivot_table(index="timestamp", columns="product", values="mid_price", aggfunc="first")
        p = p[[U] + [c for c in VOU if c in p.columns]].sort_index()
        for ts in p.index[::STEP]:
            row = p.loc[ts]
            S = float(row[U])
            T = t_y(day, int(ts))
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
                s = iv(m, S, K, T)
                if s is None:
                    continue
                xs.append(math.log(K / S) / srt)
                ys.append(s)
            if len(xs) < 6:
                continue
            a, b, c = [float(x) for x in np.polyfit(np.asarray(xs), np.asarray(ys), 2)]

            def hat(x: float) -> float:
                return a * x * x + b * x + c

            n += 1
            for k in WING:
                v = f"VEV_{k}"
                if v not in row.index or pd.isna(row[v]):
                    continue
                m = float(row[v])
                intr = max(S - k, 0.0)
                if m <= 0 or (m - intr) / m < MIN_TV:
                    continue
                s2 = iv(m, S, k, T)
                if s2 is None:
                    continue
                mt = math.log(k / S) / srt
                r = s2 - hat(mt)
                sp = spread_raw(df0, day, int(ts), v)
                if sp is None:
                    continue
                if r > RICH:
                    w_rich.append(sp)
                if r < CHEAP:
                    w_cheap.append(sp)

    def pct(arr: list[int], p: float) -> float:
        if not arr:
            return float("nan")
        return float(np.percentile(np.asarray(arr, dtype=float), p))

    pay = {
        "method": "Subsample every 30 ts; v20-style smile; spread = min ask - max bid (first level) from same CSV row as mids",
        "n_smile_fits": n,
        "n_spreads_when_rich_short": len(w_rich),
        "spread_when_rich_p50": pct(w_rich, 50),
        "n_spreads_when_cheap_long": len(w_cheap),
        "spread_when_cheap_p50": pct(w_cheap, 50),
    }
    OUT.write_text(json.dumps(pay, indent=2), encoding="utf-8")
    print(OUT.read_text())


if __name__ == "__main__":
    main()
