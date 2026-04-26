#!/usr/bin/env python3
"""
Two-hop chain Mark 55 -> Mark 01 -> Mark 22 (Phase 1 graph motif): tape-only extract forwards.

Definition (matches r4_phase1_counterparty_analysis.py):
  Leg1: trade with buyer=Mark 55, seller=Mark 01 at timestamp t1
  Leg2: later trade with buyer=Mark 01, seller=Mark 22 at t2 with t1 < t2 <= t1 + 5000 (same day)
  Outcome: dm_ex_kH = mid_extract(t2 + H*100) - mid_extract(t2) for H in {5,20,100}

Outputs:
- r4_p1_twohop_m55_m01_m22_events.csv — one row per completed chain (t1,t2,sym1,sym2,dm_ex_k*)
- r4_p1_twohop_m55_m01_m22_summary.json — pooled and per-day means, n, t-stats
"""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
TRADE_GLOB = os.path.join("Prosperity4Data", "ROUND_4", "trades_round_4_day_{d}.csv")
PRICE_GLOB = os.path.join("Prosperity4Data", "ROUND_4", "prices_round_4_day_{d}.csv")
DAYS = (1, 2, 3)
WIN = 5000
K_LIST = (5, 20, 100)
OUT_CSV = os.path.join(HERE, "analysis_outputs", "r4_p1_twohop_m55_m01_m22_events.csv")
OUT_JSON = os.path.join(HERE, "analysis_outputs", "r4_p1_twohop_m55_m01_m22_summary.json")


def load_prices(day: int) -> pd.DataFrame:
    p = pd.read_csv(PRICE_GLOB.format(d=day), sep=";")
    p["day"] = day
    return p


def load_trades(day: int) -> pd.DataFrame:
    t = pd.read_csv(TRADE_GLOB.format(d=day), sep=";")
    t["day"] = day
    return t


def mid_lookup(prices: pd.DataFrame) -> dict[tuple[int, str], dict[int, float]]:
    out: dict[tuple[int, str], dict[int, float]] = {}
    for (day, prod), g in prices.groupby(["day", "product"]):
        out[(int(day), str(prod))] = dict(zip(g["timestamp"].astype(int), g["mid_price"].astype(float)))
    return out


def mid_at(L: dict, day: int, prod: str, ts: int) -> float | None:
    d = L.get((day, prod))
    if not d:
        return None
    return d.get(int(ts))


def mid_fwd(L: dict, day: int, prod: str, ts: int, k: int) -> float | None:
    return mid_at(L, day, prod, int(ts) + int(k) * 100)


def stat(s: pd.Series) -> dict[str, Any]:
    x = s.dropna()
    n = int(len(x))
    if n < 2:
        return {"n": n, "mean": float("nan"), "t": float("nan")}
    m = float(x.mean())
    sd = float(x.std(ddof=1))
    t = m / (sd / np.sqrt(n)) if sd > 1e-12 else float("nan")
    return {"n": n, "mean": m, "t": t}


def main() -> None:
    L: dict[tuple[int, str], dict[int, float]] = {}
    for d in DAYS:
        L.update(mid_lookup(load_prices(d)))

    events: list[dict[str, Any]] = []
    for day in DAYS:
        tday = load_trades(day).sort_values("timestamp").reset_index(drop=True)
        arr = tday.to_dict("records")
        for i in range(len(arr) - 1):
            t1 = int(arr[i]["timestamp"])
            b1 = str(arr[i]["buyer"]) if pd.notna(arr[i]["buyer"]) else ""
            s1 = str(arr[i]["seller"]) if pd.notna(arr[i]["seller"]) else ""
            if b1 != "Mark 55" or s1 != "Mark 01":
                continue
            sym1 = str(arr[i]["symbol"])
            for j in range(i + 1, min(i + 200, len(arr))):
                t2 = int(arr[j]["timestamp"])
                if t2 <= t1 or t2 - t1 > WIN:
                    break
                b2 = str(arr[j]["buyer"]) if pd.notna(arr[j]["buyer"]) else ""
                s2 = str(arr[j]["seller"]) if pd.notna(arr[j]["seller"]) else ""
                if b2 != "Mark 01" or s2 != "Mark 22":
                    continue
                sym2 = str(arr[j]["symbol"])
                ex0 = mid_at(L, day, "VELVETFRUIT_EXTRACT", t2)
                rec: dict[str, Any] = {
                    "day": day,
                    "t1": t1,
                    "t2": t2,
                    "dt": t2 - t1,
                    "sym_leg1": sym1,
                    "sym_leg2": sym2,
                }
                for k in K_LIST:
                    exk = mid_fwd(L, day, "VELVETFRUIT_EXTRACT", t2, k)
                    rec[f"dm_ex_k{k}"] = (exk - ex0) if ex0 is not None and exk is not None else np.nan
                events.append(rec)

    ev = pd.DataFrame(events)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    ev.to_csv(OUT_CSV, index=False)

    summary: dict[str, Any] = {
        "chain": "Mark 55->Mark 01->Mark 22",
        "window_ts": WIN,
        "n_events": int(len(ev)),
        "pooled": {},
        "by_day": {},
    }
    for k in K_LIST:
        col = f"dm_ex_k{k}"
        summary["pooled"][col] = stat(ev[col])
    for d, g in ev.groupby("day"):
        summary["by_day"][str(int(d))] = {f"dm_ex_k{k}": stat(g[f"dm_ex_k{k}"]) for k in K_LIST}

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
